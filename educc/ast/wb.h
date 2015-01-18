#pragma once

typedef unsigned int uint;

template<typename T>
struct vec3 {
    T x, y, z;
    vec3();
    vec3(T x_, T y_ = 1, T z_ = 1);
};

typedef vec3<uint> uint3;
typedef vec3<int> dim3;
dim3 gridDim;
dim3 blockDim;
uint3 threadIdx;
uint3 blockIdx;

struct driver_t {
    driver_t(dim3, dim3);
    void invoke_kernel(...);
};


#define __shared__ __attribute__((annotate("__shared__")))
#define __global__ __attribute__((annotate("__global__")))
#define __device__ __attribute__((annotate("__device__")))
#define __host__ __attribute__((annotate("__host__")))
#define static __attribute__((annotate("__static__")))

void __device__ __syncthreads();
#define min(a,b) a

#include <stdlib.h>
