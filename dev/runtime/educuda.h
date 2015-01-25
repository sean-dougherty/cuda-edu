#pragma once

#define EDU_CUDA_SHARED_STORAGE static edu_thread_local

#include <educuda-api.h>
#include <eduguard.h>
#include <edupfm.h>
#include <microblanket.h>

#include <assert.h>
#include <functional>
#include <memory>
#include <thread>
#include <vector>


#define __edu_cuda_invoke_kernel(driver, x...) driver.invoke_kernel([=]{x;})

namespace edu {
    namespace cuda {

        edu_thread_local uint3 blockIdx;
        edu_thread_local uint3 threadIdx;
        edu_thread_local guard::ptr_guard_t<char> dynamic_shared;

#define __edu_cuda_get_dynamic_shared() dynamic_shared

        uint linearize(const dim3 &dim, uint3 idx) {
            return (dim.x*dim.y)*idx.z + idx.y*dim.x + idx.x;
        }

        uint3 delinearize(const dim3 &dim, uint idx) {
            uint z = idx / (dim.x*dim.y);
            uint remain = idx % (dim.x*dim.y);
            uint y = remain / dim.x;
            uint x = remain % dim.x;
            return {x, y, z};
        }

        const size_t Cuda_Thread_Stack_Size = 8*1024;

        //------------------------------------------------------------
        //---
        //--- STRUCT cuda_thread_t
        //---
        //------------------------------------------------------------
        struct cuda_thread_t : microblanket::fiber_t<Cuda_Thread_Stack_Size> {
            uint3 idx;

            enum status_t {
                Birth,
                Spawn,
                Run,
                Sync,
                SyncWarp,
                Exit
            } status = Birth;

            static edu_thread_local cuda_thread_t *current;

#define __syncthreads() edu::cuda::cuda_thread_t::current->sync()

            void set_current() {
                threadIdx = idx;
                current = this;
            }

            void sync() {
                assert(status == Run);
                status = Sync;
                yield();
            }

            void sync_warp() {
                assert(status == Run);
                status = SyncWarp;
                yield();
            }

            void resume() {
                assert( (status == Sync) || (status == SyncWarp) || (status == Spawn));
                status = Run;
                set_current();
                microblanket::fiber_t<Cuda_Thread_Stack_Size>::resume();
            }

            void run(uint3 idx_, function<void()> enter_kernel) {
                idx = idx_;
                status = Spawn;
                yield();
                status = Run;
                set_current();
                enter_kernel();
                status = Exit;
            }
        };        

        edu_thread_local cuda_thread_t *cuda_thread_t::current = nullptr;

        //------------------------------------------------------------
        //---
        //--- STRUCT block_state_t
        //---
        //------------------------------------------------------------
        struct block_state_t {
            unsigned n;
            unsigned nsync;
            unsigned nexit;

            block_state_t(unsigned n_)
            : n(n_) {
                reset();
            }

            void reset() {
                nsync = nexit = 0;
            }

            void update(cuda_thread_t *t) {
                switch(t->status) {
                case cuda_thread_t::Sync: nsync++; break;
                case cuda_thread_t::Exit: nexit++; break;
                default: abort();
                }
            }

            bool is_done() {
                if(nexit == n)
                    return true;
                if(nsync == n)
                    return false;
                if(nexit != 0) {
                    edu_err("Some, but not all, threads have exited the kernel!");
                }
                if(nsync != 0) {
                    edu_err("Some, but not all, threads have synced!");
                }
                edu_panic("n=" << n << ", nsync=" << nsync << ", nexit=" << nexit);
            }
        };

        struct driver_t {
            dim3 gridDim;
            dim3 blockDim;
            unsigned int dynamic_shared_size;

            driver_t(dim3 gridDim_,
                     dim3 blockDim_,
                     unsigned int dynamic_shared_size_ = 0)
            : gridDim(gridDim_)
            , blockDim(blockDim_)
            , dynamic_shared_size(dynamic_shared_size_){
            }

            void invoke_kernel(function<void ()> enter_kernel) {

                if(cudaSuccess != check_kernel_config(gridDim, blockDim)) {
                    return;
                }

                mem::set_space(mem::MemorySpace_Device);
                guard::set_write_callback([](){cuda_thread_t::current->sync_warp();});

                const unsigned nos_threads = pfm::get_thread_count();

                char *all_dynamic_shared[nos_threads];
                if(dynamic_shared_size) {
                    for(int i = 0; i < nos_threads; i++) {
                        all_dynamic_shared[i] = (char *)mem::alloc(mem::MemorySpace_Device,
                                                                   dynamic_shared_size);
                        edu_errif(!all_dynamic_shared[i]);
                    }
                }
                
                cuda::gridDim = gridDim;
                cuda::blockDim = blockDim;
                
                const unsigned nblocks = gridDim.x * gridDim.y * gridDim.z;
                const unsigned ncuda_threads = blockDim.x * blockDim.y * blockDim.z;
                const unsigned blocks_per_thread = nblocks / ncuda_threads;

                vector<unique_ptr<thread>> threads;
                // I would just use OpenMP here, but not supported by clang yet.
                for(unsigned ithread = 0;
                    ithread < nos_threads;
                    ithread++) {

                    unsigned iblock_start = ithread * blocks_per_thread;
                    unsigned iblock_end = (ithread == nos_threads - 1) ? nblocks : iblock_start + nblocks;

                    char *thread_dynamic_shared;
                    if(dynamic_shared_size) {
                        thread_dynamic_shared = all_dynamic_shared[ithread];
                    } else {
                        thread_dynamic_shared = nullptr;
                    }

                    threads.push_back(
                        unique_ptr<thread>(
                            new thread([ithread,
                                        iblock_start,
                                        iblock_end,
                                        ncuda_threads,
                                        thread_dynamic_shared,
                                        enter_kernel]() {

                                           dynamic_shared = thread_dynamic_shared;

                                           for(size_t iblock = iblock_start; iblock < iblock_end; iblock++) {
                                               microblanket::blanket_t<cuda_thread_t> cuda_threads{ncuda_threads};
                                               blockIdx = delinearize(cuda::gridDim, iblock);
                                               block_state_t block_state{ncuda_threads};

                                               for(uint3 tIdx = {0,0,0}; tIdx.x < cuda::blockDim.x; tIdx.x++) {
                                                   for(tIdx.y = 0; tIdx.y < cuda::blockDim.y; tIdx.y++) {
                                                       for(tIdx.z = 0; tIdx.z < cuda::blockDim.z; tIdx.z++) {
                                                           cuda_threads.spawn(
                                                               [tIdx, enter_kernel]
                                                               (cuda_thread_t *cuda_thread) {
                                                                   cuda_thread->run(tIdx, enter_kernel);
                                                               });
                                                       }
                                                   }
                                               }

                                               do {
                                                   block_state.reset();
                                                   for(unsigned iwarp_start = 0;
                                                       iwarp_start < ncuda_threads;
                                                       iwarp_start += EDU_CUDA_WARP_SIZE) {

                                                       bool in_sync_warp;
                                                       bool first = true;
                                                       do {
                                                           in_sync_warp = false;
                                                           unsigned iwarp_end = min(ncuda_threads, iwarp_start + EDU_CUDA_WARP_SIZE);
                                                           for(unsigned ithread = iwarp_start;
                                                               ithread < iwarp_end;
                                                               ithread++) {

                                                               cuda_thread_t *t = cuda_threads.get_fiber(ithread);
                                                               if(first || (t->status == cuda_thread_t::SyncWarp)) {
                                                                   t->resume();
                                                                   in_sync_warp |= (t->status == cuda_thread_t::SyncWarp);
                                                               }
                                                           }
                                                           first = false;
                                                       } while(in_sync_warp);
                                                   }

                                                   for(unsigned ithread = 0; ithread < ncuda_threads; ithread++) {
                                                       block_state.update( cuda_threads.get_fiber(ithread) );
                                                   }
                                               } while(!block_state.is_done());
                                           }
                                       })));
                }

                for(auto &t: threads) {
                    t->join();
                }

                if(dynamic_shared_size) {
                    for(int i = 0; i < nos_threads; i++) {
                        mem::dealloc(mem::MemorySpace_Device, all_dynamic_shared[i]);
                    }
                }

                mem::set_space(mem::MemorySpace_Host);
                guard::clear_write_callback();
            }
        };

    }
}
