#pragma once

#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define ERROR 0
#define TRACE 0
#define GPU 0
#define wbLog(x...)
#define wbTime_start(x...)
#define wbTime_stop(x...)
#define wbSolution(x...)

typedef void *wbArg_t;
wbArg_t wbArg_read(...);
char *wbArg_getInputFile(...);
void *wbImport(...);

typedef void *wbImage_t;
wbImage_t wbImage_new(...);
void wbImage_delete(...);
int wbImage_getWidth(...);
int wbImage_getHeight(...);
int wbImage_getChannels(...);
float *wbImage_getData(...);
wbImage_t wbImport(...);

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
    int maxThreadsPerMultiProcessor;
    int multiProcessorCount;
};

#define __shared__ __attribute__((annotate("__shared__")))
#define __global__ __attribute__((annotate("__global__")))
#define __device__ __attribute__((annotate("__device__")))
#define __host__ __attribute__((annotate("__host__")))
#define static __attribute__((annotate("__static__")))

enum cudaError_t {
    cudaSuccess
};

void __device__ __syncthreads();
cudaError_t cudaGetDeviceCount(...);
cudaError_t cudaGetDeviceProperties(...);
cudaError_t cudaMalloc(...);
cudaError_t cudaFree(...);
cudaError_t cudaMemcpy(...);
cudaError_t cudaMemset(...);
cudaError_t cudaDeviceSynchronize(...);
cudaError_t cudaThreadSynchronize(...);
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0
#define min(a,b) a
#define max(a,b) a

struct driver_t {
    driver_t(dim3, dim3);
    void invoke_kernel(...);
};
