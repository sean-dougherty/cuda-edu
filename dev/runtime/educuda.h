#pragma once

#define EDU_CUDA_SHARED_STORAGE static edu_thread_local

#include <educuda-api.h>
#include <eduguard.h>
#include <edupfm.h>

#include <assert.h>
#include <functional>
#include <memory>
#include <thread>
#include <vector>


#if !defined(EDU_CUDA_FIBERS_OS_THREADS_COUNT)
    #define EDU_CUDA_FIBERS_OS_THREADS_COUNT 4
#endif

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

        struct fiber_t {
            uint3 idx;
            function<void()> enter_kernel;
            pfm::fiber_context_t ctx;
            pfm::fiber_context_t ctx_main;
            char stack[pfm::Min_Stack_Size];
            enum status_t {
                Birth,
                Spawn,
                Run,
                Sync,
                SyncWarp,
                Exit
            } status = Birth;

            static edu_thread_local fiber_t *current;

#define __syncthreads() edu::cuda::fiber_t::current->sync()

            void set_current() {
                threadIdx = idx;
                current = this;
            }

            void sync() {
                assert(status == Run);
                status = Sync;
                edu_errif(!pfm::switch_fiber_context(&ctx, &ctx_main));
            }

            void sync_warp() {
                assert(status == Run);
                status = SyncWarp;
                edu_errif(!pfm::switch_fiber_context(&ctx, &ctx_main));
            }

            void resume() {
                assert( (status == Sync) || (status == SyncWarp) || (status == Spawn));
                status = Run;
                set_current();
                edu_errif(!pfm::switch_fiber_context(&ctx_main, &ctx));
            }

            void run() {
                set_current();
                enter_kernel();
            }

            static void __run(fiber_t *thiz) {
                thiz->status = Run;
                thiz->run();
                thiz->status = Exit;
            }

            void spawn() {
                status = Spawn;

                pfm::init_fiber_context(&ctx,
                                        &ctx_main,
                                        stack,
                                        sizeof(stack),
                                        (pfm::fiber_context_entry_func_t)__run,
                                        this);
            }

        fiber_t(uint3 idx_,
                function<void()> enter_kernel_)
        : idx(idx_)
        , enter_kernel(enter_kernel_)
                , status(Birth) {
        }

            ~fiber_t() {
                pfm::dispose_fiber_context(ctx);
                pfm::dispose_fiber_context(ctx_main);
            }
        };        

        edu_thread_local fiber_t *fiber_t::current = nullptr;

        struct fibers_block_t {
            unsigned n;
            unsigned nsync;
            unsigned nexit;

        fibers_block_t(unsigned n_)
        : n(n_) {
            reset();
        }

            void reset() {
                nsync = nexit = 0;
            }

            void update(fiber_t f) {
                switch(f.status) {
                case fiber_t::Sync: nsync++; break;
                case fiber_t::Exit: nexit++; break;
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

            template<typename... T>
            void invoke_kernel(void (*kernel)(T... args), T... args) {

                if(cudaSuccess != check_kernel_config(gridDim, blockDim)) {
                    return;
                }

                auto enter_kernel = [=]() {
                    kernel(args...);
                };

                mem::set_space(mem::MemorySpace_Device);
                guard::set_write_callback([](){fiber_t::current->sync_warp();});

                char *all_dynamic_shared[EDU_CUDA_FIBERS_OS_THREADS_COUNT];
                if(dynamic_shared_size) {
                    for(int i = 0; i < EDU_CUDA_FIBERS_OS_THREADS_COUNT; i++) {
                        all_dynamic_shared[i] = (char *)mem::alloc(mem::MemorySpace_Device, dynamic_shared_size);
                        edu_errif(!all_dynamic_shared[i]);
                    }
                }
                
                cuda::gridDim = gridDim;
                cuda::blockDim = blockDim;
                
                const unsigned nblocks = gridDim.x * gridDim.y * gridDim.z;
                const unsigned ncuda_threads = blockDim.x * blockDim.y * blockDim.z;

                const unsigned blocks_per_thread = nblocks / ncuda_threads;

                vector<unique_ptr<thread>> threads;
                // I would just use OpenMP here, but Apple.
                for(unsigned ithread = 0;
                    ithread < EDU_CUDA_FIBERS_OS_THREADS_COUNT;
                    ithread++) {

                    unsigned iblock_start = ithread * blocks_per_thread;
                    unsigned iblock_end = (ithread == EDU_CUDA_FIBERS_OS_THREADS_COUNT - 1) ? nblocks : iblock_start + nblocks;

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
                                               blockIdx = delinearize(cuda::gridDim, iblock);
                                               vector<fiber_t> fibers;
                                               fibers.reserve(ncuda_threads);
                            
                                               fibers_block_t fblock{ncuda_threads};

                                               for(uint3 tIdx = {0,0,0}; tIdx.x < cuda::blockDim.x; tIdx.x++) {
                                                   for(tIdx.y = 0; tIdx.y < cuda::blockDim.y; tIdx.y++) {
                                                       for(tIdx.z = 0; tIdx.z < cuda::blockDim.z; tIdx.z++) {

                                                           fibers.emplace_back(tIdx, enter_kernel);
                                                           fiber_t &f = fibers.back();
                                                           f.spawn();
                                                       }
                                                   }
                                               }

                                               do {
                                                   fblock.reset();
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

                                                               fiber_t &f = fibers[ithread];
                                                               if(first || (f.status == fiber_t::SyncWarp)) {
                                                                   f.resume();
                                                                   in_sync_warp |= (f.status == fiber_t::SyncWarp);
                                                               }
                                                           }
                                                           first = false;
                                                       } while(in_sync_warp);
                                                   }

                                                   for(fiber_t &f: fibers) {
                                                       fblock.update(f);
                                                   }
                                               } while(!fblock.is_done());
                                           }
                                       })));
                }

                for(auto &t: threads) {
                    t->join();
                }

                if(dynamic_shared_size) {
                    for(int i = 0; i < EDU_CUDA_FIBERS_OS_THREADS_COUNT; i++) {
                        mem::dealloc(mem::MemorySpace_Device, all_dynamic_shared[i]);
                    }
                }

                mem::set_space(mem::MemorySpace_Host);
                guard::clear_write_callback();
            }
        };

    }
}
