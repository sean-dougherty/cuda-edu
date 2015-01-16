#pragma once

#include <condition_variable>
#include <thread>
#include <mutex>
#include <vector>

#include <educuda-api.h>

#define __shared__ static

namespace edu {
    namespace cuda {

        thread_local uint3 threadIdx;

        class Barrier {
        private:
            mutex _mutex;
            condition_variable _cv;
            size_t _nthreads;
            size_t _count[2];
            size_t _idx;
            static mutex all_mutex;
            static vector<Barrier *> *all;

        public:
            Barrier() {
                std::unique_lock<std::mutex> lock{all_mutex};
                if(!all) {
                    all = new vector<Barrier *>();
                }
                all->push_back(this);
                reset();
            }

            static void reset_all() {
                std::unique_lock<std::mutex> lock{all_mutex};
                if(all) {
                    for(Barrier *b: *all)
                        b->reset();
                }
            }

            void reset() {
                _nthreads = blockDim.x * blockDim.y * blockDim.z;
                _count[0] = _nthreads;
                _idx = 0;
            }
                
            void sync() {
                std::unique_lock<std::mutex> lock{_mutex};
                int idx = _idx;

                if(--_count[idx] == 0) {
                    _cv.notify_all();
                    _idx = (idx + 1) % 2;
                    _count[_idx] = _nthreads;
                } else {
                    _cv.wait(lock, [this, idx] { return _count[idx] == 0; });
                }
            }
        };
        mutex Barrier::all_mutex;
        vector<Barrier *> *Barrier::all = nullptr;

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

                mem::set_space(mem::MemorySpace_Device);
                
                cuda::gridDim = gridDim;
                cuda::blockDim = blockDim;
                
                for(blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {
                    for(blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
                        for(blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
                            
                            Barrier::reset_all();

                            vector<unique_ptr<thread>> threads;
                            for(uint3 tIdx = {0,0,0}; tIdx.x < blockDim.x; tIdx.x++) {
                                for(tIdx.y = 0; tIdx.y < blockDim.y; tIdx.y++) {
                                    for(tIdx.z = 0; tIdx.z < blockDim.z; tIdx.z++) {
                                        
                                        unique_ptr<thread> t(
                                            new thread([=] () {
                                                    threadIdx = tIdx;
                                                    kernel(args...);
                                                    return 0;
                                                }));
                                        threads.push_back(move(t));
                                    }
                                }
                            }

                            for(auto &t: threads) {
                                t->join();
                            }

                        }
                    }
                }

                mem::set_space(mem::MemorySpace_Host);
            }
        };

#define __syncthreads() {                        \
            static Barrier barrier_##__LINE__; barrier_##__LINE__.sync(); \
        }
    }
}
