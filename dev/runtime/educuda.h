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

#if EDU_CUDA_COMPILE_PASS == 0
    // For the first pass, we just define globals.
    edu::cuda::uint3 blockIdx;
    edu::cuda::uint3 threadIdx;
#endif

#define __edu_cuda_invoke_kernel(driver, x...) driver.invoke_kernel([=]{x;})
#define __edu_cuda_get_dynamic_shared() dynamic_shared
#define __edu_cuda_decl_fls                                             \
    uint3 blockIdx = edu::cuda::current_cuda_thread()->blockIdx;     \
    uint3 threadIdx = edu::cuda::current_cuda_thread()->threadIdx;

#define __syncthreads() edu::cuda::current_cuda_thread()->sync()

namespace edu {
    namespace cuda {

        edu_thread_local guard::ptr_guard_t<char> dynamic_shared;

        uint linearize(const dim3 &dim, const uint3 &idx) {
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
            uint3 blockIdx;
            uint3 threadIdx;

            enum status_t {
                Birth,
                Spawn,
                Run,
                Sync,
                SyncWarp,
                Exit
            } status = Birth;

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
                microblanket::fiber_t<Cuda_Thread_Stack_Size>::resume();
            }

            void run(uint3 blockIdx_, uint3 threadIdx_, function<void()> enter_kernel) {
                blockIdx = blockIdx_;
                threadIdx = threadIdx_;
                status = Spawn;
                yield();
                status = Run;
                enter_kernel();
                status = Exit;
            }
        };        

        cuda_thread_t *current_cuda_thread() {
            return microblanket::blanket_t<cuda_thread_t>::current_fiber();
        }

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

            void update(cuda_thread_t &t) {
                switch(t.status) {
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

        //------------------------------------------------------------
        //---
        //--- STRUCT driver_t
        //---
        //------------------------------------------------------------
        struct driver_t {
            dim3 gridDim;
            dim3 blockDim;
            unsigned int dynamic_shared_size;

            vector<char *> all_dynamic_shared;

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

                cuda::gridDim = gridDim;
                cuda::blockDim = blockDim;

                mem::set_space(mem::MemorySpace_Device);
                guard::set_write_callback([](){current_cuda_thread()->sync_warp();});

                const unsigned nos_threads = pfm::get_thread_count();                
                const unsigned nblocks = gridDim.x * gridDim.y * gridDim.z;
                const unsigned ncuda_threads = blockDim.x * blockDim.y * blockDim.z;
                const unsigned blocks_per_osthread = nblocks / nos_threads;

                if(dynamic_shared_size) {
                    all_dynamic_shared.resize(nos_threads);
                    for(unsigned ithread = 0; ithread < nos_threads; ithread++) {
                        all_dynamic_shared[ithread] =
                            (char *)mem::alloc(mem::MemorySpace_Device,
                                               dynamic_shared_size);
                    }
                }

                vector<unique_ptr<thread>> threads;
                // I would just use OpenMP here, but not supported by clang yet.
                for(unsigned ithread = 0; ithread < nos_threads; ithread++) {

                    unsigned iblock_start = ithread * blocks_per_osthread;
                    unsigned iblock_end = (ithread == nos_threads - 1) ? nblocks : iblock_start + nblocks;

                    auto osthread_task = [=]() {
                        if(dynamic_shared_size) {
                            dynamic_shared = all_dynamic_shared[ithread];
                        }
                        execute_blocks(iblock_start,
                                       iblock_end,
                                       ncuda_threads,
                                       enter_kernel);
                        if(dynamic_shared_size) {
                            dynamic_shared = nullptr;
                        }
                    };

                    threads.push_back(unique_ptr<thread>(new thread(osthread_task)));
                }

                for(auto &t: threads) {
                    t->join();
                }

                if(dynamic_shared_size) {
                    for(unsigned ithread = 0; ithread < nos_threads; ithread++) {
                        mem::dealloc(mem::MemorySpace_Device, all_dynamic_shared[ithread]);
                        all_dynamic_shared[ithread] = nullptr;
                    }
                }


                mem::set_space(mem::MemorySpace_Host);
                guard::clear_write_callback();
            }

            void execute_blocks(unsigned iblock_start,
                                unsigned iblock_end,
                                unsigned ncuda_threads,
                                function<void()> enter_kernel) {
                microblanket::blanket_t<cuda_thread_t> cuda_threads{ncuda_threads};

                for(size_t iblock = iblock_start; iblock < iblock_end; iblock++) {
                    uint3 blockIdx = delinearize(cuda::gridDim, iblock);
                    block_state_t block_state{ncuda_threads};

                    cuda_threads.clear();

                    for(uint3 tIdx = {0,0,0}; tIdx.x < cuda::blockDim.x; tIdx.x++) {
                        for(tIdx.y = 0; tIdx.y < cuda::blockDim.y; tIdx.y++) {
                            for(tIdx.z = 0; tIdx.z < cuda::blockDim.z; tIdx.z++) {
                                cuda_threads.spawn(
                                    [blockIdx, tIdx, enter_kernel]
                                    (cuda_thread_t *cuda_thread) {
                                        cuda_thread->run(blockIdx, tIdx, enter_kernel);
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

                                    cuda_thread_t &t = cuda_threads[ithread];
                                    if(first || (t.status == cuda_thread_t::SyncWarp)) {
                                        t.resume();
                                        in_sync_warp |= (t.status == cuda_thread_t::SyncWarp);
                                    }
                                }
                                first = false;
                            } while(in_sync_warp);
                        }

                        for(auto &t: cuda_threads) {
                            block_state.update(t);
                        }
                    } while(!block_state.is_done());
                }
            }
        };

    }
}
