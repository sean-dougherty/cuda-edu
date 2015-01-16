#pragma once

#include <educuda-api.h>
#include <ucontext.h>

#include <assert.h>
#include <functional>
#include <vector>

#define __shared__ static

namespace edu {
    namespace cuda {

        uint3 threadIdx;
#define handle_error(msg)                               \
        do { perror(msg); exit(EXIT_FAILURE); } while (0)

        typedef void (*ucontext_func_t)();

        const size_t SIGNATURE_PREFIX = 0xdeadbeef;
        const size_t SIGNATURE_SUFFIX = 0xabcd1234;

        struct fiber_t {
            uint3 idx;
            function<void()> enter_kernel;
            ucontext_t uctx;
            ucontext_t uctx_main;
            char stack[4096];
            enum status_t {
                Birth,
                Spawn,
                Run,
                Sync,
                Exit
            } status = Birth;

            static fiber_t *current;

#define __syncthreads() edu::cuda::fiber_t::current->sync()

            void set_current() {
                threadIdx = idx;
                current = this;
            }

            void sync() {
                assert(status == Run);
                status = Sync;
                swapcontext(&uctx, &uctx_main);
            }

            void resume() {
                assert(status == Sync);
                status = Run;
                set_current();
                swapcontext(&uctx_main, &uctx);
            }

            void run() {
                set_current();
                enter_kernel();
            }

            static void __run(size_t signature_prefix,
                              fiber_t *thiz,
                              size_t signature_suffix) {
                assert(signature_prefix == SIGNATURE_PREFIX);
                assert(signature_suffix == SIGNATURE_SUFFIX);

                thiz->status = Run;
                thiz->run();
                thiz->status = Exit;
            }

            void spawn() {
                status = Spawn;

                uctx.uc_stack.ss_sp = stack;
                uctx.uc_stack.ss_size = sizeof(stack);
                uctx.uc_link = &uctx_main;
                makecontext(&uctx, (ucontext_func_t)__run,
                            3, SIGNATURE_PREFIX, this, SIGNATURE_SUFFIX);
                if(swapcontext(&uctx_main, &uctx) == -1)
                    handle_error("swapcontext");
            }

            fiber_t(uint3 idx_,
                    function<void()> enter_kernel_)
            : idx(idx_)
            , enter_kernel(enter_kernel_)
                , status(Birth) {

                if(getcontext(&uctx) == -1)
                    handle_error("getcontext");
            }
        };        

        fiber_t *fiber_t::current = nullptr;

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

            driver_t(dim3 gridDim_,
                     dim3 blockDim_)
            : gridDim(gridDim_)
            , blockDim(blockDim_) {
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
                
                cuda::gridDim = gridDim;
                cuda::blockDim = blockDim;
                
                size_t nthreads = blockDim.x * blockDim.y * blockDim.z;

                for(blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {
                    for(blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
                        for(blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
                            
                            vector<fiber_t> fibers;
                            fibers.reserve(nthreads);
                            
                            fibers_block_t fblock{nthreads};

                            for(uint3 tIdx = {0,0,0}; tIdx.x < blockDim.x; tIdx.x++) {
                                for(tIdx.y = 0; tIdx.y < blockDim.y; tIdx.y++) {
                                    for(tIdx.z = 0; tIdx.z < blockDim.z; tIdx.z++) {

                                        fibers.emplace_back(tIdx, enter_kernel);
                                        fiber_t &f = fibers.back();
                                        f.spawn();
                                        fblock.update(f);
                                    }
                                }
                            }

                            while(!fblock.is_done()) {
                                fblock.reset();
                                for(fiber_t &f: fibers) {
                                    f.resume();
                                    fblock.update(f);
                                }
                            }

                        }
                    }
                }

                mem::set_space(mem::MemorySpace_Host);
            }
        };

    }
}
