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

struct float1 { float x; };
float1 make_float1(float x) { return {x}; }
struct float2 { float x, y; };
float2 make_float2(float x, float y) { return {x,y}; }
struct float3 { float x, y, z; };
float3 make_float3(float x, float y, float z) { return {x,y,z}; }
struct float4 { float x, y, z, w; };
float4 make_float4(float x, float y, float z, float w) { return {x,y,z,w}; }

struct char1 { char x; };
char1 make_char1(char x) { return {x}; }
struct char2 { char x, y; };
char2 make_char2(char x, char y) { return {x,y}; }
struct char3 { char x, y, z; };
char3 make_char3(char x, char y, char z) { return {x,y,z}; }
struct char4 { char x, y, z, w; };
char4 make_char4( char x,  char y,  char z,  char w) { return {x,y,z,w}; }

struct uchar1 { unsigned char x; };
uchar1 make_uchar1(unsigned char x) { return {x}; }
struct uchar2 { unsigned char x, y; };
uchar2 make_char2(unsigned char x, unsigned char y) { return {x,y}; }
struct uchar3 { unsigned char x, y, z; };
uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) { return {x,y,z}; }
struct uchar4 { unsigned char x, y, z, w; };
uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { return {x,y,z,w}; }

struct short1 { short x; };
short1 make_short1(short x) { return {x}; }
struct short2 { short x, y; };
short2 make_short2(short x, short y) { return {x,y}; }
struct short3 { short x, y, z; };
short3 make_short3(short x, short y, short z) { return {x,y,z}; }
struct short4 { short x, y, z, w; };
short4 make_short4( short x,  short y,  short z,  short w) { return {x,y,z,w}; }

struct ushort1 { unsigned short x; };
ushort1 make_ushort1(unsigned short x) { return {x}; }
struct ushort2 { unsigned short x, y; };
ushort2 make_short2(unsigned short x, unsigned short y) { return {x,y}; }
struct ushort3 { unsigned short x, y, z; };
ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) { return {x,y,z}; }
struct ushort4 { unsigned short x, y, z, w; };
ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) { return {x,y,z,w}; }

struct int1 { int x; };
int1 make_int1(int x) { return {x}; }
struct int2 { int x, y; };
int2 make_int2(int x, int y) { return {x,y}; }
struct int3 { int x, y, z; };
int3 make_int3(int x, int y, int z) { return {x,y,z}; }
struct int4 { int x, y, z, w; };
int4 make_int4( int x,  int y,  int z,  int w) { return {x,y,z,w}; }

struct uint1 { unsigned int x; };
uint1 make_uint1(unsigned int x) { return {x}; }
struct uint2 { unsigned int x, y; };
uint2 make_int2(unsigned int x, unsigned int y) { return {x,y}; }
struct uint3 { unsigned int x, y, z; };
uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return {x,y,z}; }
struct uint4 { unsigned int x, y, z, w; };
uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return {x,y,z,w}; }

struct long1 { long x; };
long1 make_long1(long x) { return {x}; }
struct long2 { long x, y; };
long2 make_long2(long x, long y) { return {x,y}; }
struct long3 { long x, y, z; };
long3 make_long3(long x, long y, long z) { return {x,y,z}; }
struct long4 { long x, y, z, w; };
long4 make_long4( long x,  long y,  long z,  long w) { return {x,y,z,w}; }

struct ulong1 { unsigned long x; };
ulong1 make_ulong1(unsigned long x) { return {x}; }
struct ulong2 { unsigned long x, y; };
ulong2 make_long2(unsigned long x, unsigned long y) { return {x,y}; }
struct ulong3 { unsigned long x, y, z; };
ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) { return {x,y,z}; }
struct ulong4 { unsigned long x, y, z, w; };
ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) { return {x,y,z,w}; }


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
#define __constant__ __attribute__((annotate("__constant__")))
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
cudaError_t cudaMemcpyToSymbol(...);
cudaError_t cudaMemset(...);
cudaError_t cudaDeviceSynchronize(...);
cudaError_t cudaThreadSynchronize(...);
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0
#define min(a,b) a
#define max(a,b) a

struct driver_t {
    driver_t(dim3, dim3);
    driver_t(dim3, dim3, unsigned int);
};
#define __edu_cuda_invoke_kernel(driver, x...) x
