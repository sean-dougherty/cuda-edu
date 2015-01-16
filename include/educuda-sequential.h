#pragma once

#include <educuda-api.h>

#include <vector>
#include <memory>

#define __shared__ static

namespace edu {
    namespace cuda {

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

    }
}
