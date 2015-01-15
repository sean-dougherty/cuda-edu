#pragma once

#include <edumem.h>
#include <edupfm.h>

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__

#define __shared__ static

namespace edu {
    namespace cuda {
        using namespace std;

        enum cudaError_t {
            cudaSuccess,
            Not_Enough_Memory,
            Invalid_Device,
            Invalid_Grid_Dim,
            Invalid_Block_Dim
        };
        thread_local cudaError_t last_error = cudaSuccess;
#define ret_err(err) {last_error = err; return err;}

        const char *cudaGetErrorString(cudaError_t err) {
            switch(err) {
            case cudaSuccess:
                return "Success";
            case Not_Enough_Memory:
                return "Out of memory";
            case Invalid_Device:
                return "Invalid device number";
            case Invalid_Grid_Dim:
                return "Illegal gridDim";
            case Invalid_Block_Dim:
                return "Illegal blockDim";
            }
        }

        typedef unsigned int uint;

        template<typename T>
            struct vec3 {
                T x, y, z;

            vec3() : vec3(0) {}

            vec3(T x_, T y_ = 1, T z_ = 1)
            : x(x_), y(y_), z(z_) {
            }
            };

        typedef vec3<uint> uint3;
        typedef vec3<int> dim3;

        struct cudaDeviceProp {
            char name[256];
            int major;
            int minor;
            size_t totalGlobalMem;
            size_t totalConstMem;
            size_t sharedMemPerBlock;
            int maxThreadsPerBlock;
            int maxThreadsDim[3];
            int maxGridSize[3];
            int warpSize;
        };

        cudaError_t cudaGetDeviceCount(int *count) {
            *count = 1;
            ret_err(cudaSuccess);
        }

        cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop,
                                            int device) {
            if(device != 0)
                ret_err(Invalid_Device);

            strcpy(prop->name, "cuda-edu fake device");
            prop->major = 3;
            prop->minor = 0;
            prop->totalGlobalMem = 4294770688;
            prop->totalConstMem = 65536;
            prop->sharedMemPerBlock = 49152;
            prop->maxThreadsPerBlock = 1024;
            prop->maxThreadsDim[0] = 1024;
            prop->maxThreadsDim[1] = 1024;
            prop->maxThreadsDim[2] = 64;
            prop->maxGridSize[0] = 2147483647;
            prop->maxGridSize[1] = 65535;
            prop->maxGridSize[2] = 65535;
            prop->warpSize = 32;

            ret_err(cudaSuccess);
        }

        cudaError_t check_kernel_config(dim3 gridDim, dim3 blockDim) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            if( (gridDim.x > prop.maxGridSize[0])
                || (gridDim.y > prop.maxGridSize[1])
                || (gridDim.z > prop.maxGridSize[2]) ) {
                ret_err(Invalid_Grid_Dim);
            }

            if( (blockDim.x > prop.maxThreadsDim[0])
                || (blockDim.y > prop.maxThreadsDim[1])
                || (blockDim.z > prop.maxThreadsDim[2]) ) {
                ret_err(Invalid_Block_Dim);
            }

            if( (blockDim.x * blockDim.y * blockDim.z) > prop.maxThreadsPerBlock ) {
                ret_err(Invalid_Block_Dim);
            }

            ret_err(cudaSuccess);
        }

        cudaError_t cudaMalloc(void **ptr, size_t length) {
            void *result = mem::alloc(mem::MemorySpace_Device, length);
            if(!result)
                ret_err(Not_Enough_Memory);
            *ptr = result;
            ret_err(cudaSuccess);
        }

        cudaError_t cudaFree(void *ptr) {
            mem::dealloc(mem::MemorySpace_Device, ptr);
            ret_err(cudaSuccess);
        }

        cudaError_t cudaMallocHost(void **ptr, size_t length) {
            void *result = mem::alloc(mem::MemorySpace_Host, length);
            if(!result)
                ret_err(Not_Enough_Memory);
            *ptr = result;
            ret_err(cudaSuccess);
        }

        cudaError_t cudaFreeHost(void *ptr) {
            mem::dealloc(mem::MemorySpace_Host, ptr);
            ret_err(cudaSuccess);
        }

        enum cudaMemcpyKind {
            cudaMemcpyHostToHost = 0,
            cudaMemcpyHostToDevice = 1,
            cudaMemcpyDeviceToHost = 2,
            cudaMemcpyDeviceToDevice = 3
        };
        struct MemcpyKindSpace {
            mem::MemorySpace src;
            mem::MemorySpace dst;
        } memcpy_space[] = {
            {mem::MemorySpace_Host, mem::MemorySpace_Host},
            {mem::MemorySpace_Host, mem::MemorySpace_Device},
            {mem::MemorySpace_Device, mem::MemorySpace_Host},
            {mem::MemorySpace_Device, mem::MemorySpace_Device}
        };

        cudaError_t cudaMemcpy(void *dst,
                               const void *src,
                               size_t count,
                               cudaMemcpyKind kind) {
            mem::copy(memcpy_space[kind].dst, dst,
                      memcpy_space[kind].src, src,
                      count);
            ret_err(cudaSuccess);
        }

        cudaError_t cudaGetLastError() {
            return last_error;
        }
        cudaError_t cudaThreadSynchronize () {
            return last_error;
        }
        cudaError_t cudaDeviceSynchronize() {
            return last_error;
        }

        dim3 gridDim;
        dim3 blockDim;
        uint3 blockIdx;

#ifdef EDU_CUDA_SEQUENTIAL
        uint3 threadIdx;

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
                            
                            for(threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++) {
                                for(threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {
                                    for(threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                                        kernel(args...);
                                    }
                                }
                            }

                        }
                    }
                }

                mem::set_space(mem::MemorySpace_Host);
            }
        };

#define __syncthreads() {                                               \
            cerr << "__syncthreads() not permitted with sequential execution."<< endl; \
            cerr << "Please undefine EDU_CUDA_SEQUENTIAL." << endl;     \
            exit(1);                                                    \
        }
#else
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
#endif
    }
}
