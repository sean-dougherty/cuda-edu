#pragma once

#include <edumem.h>
#include <edupfm.h>

#define __global__
#define __device__
#define __host__
#define __constant__

#ifndef EDU_CUDA_COMPILE_PASS
    #error "EDU_CUDA_COMPILE_PASS not defined!"
#endif

#if EDU_CUDA_COMPILE_PASS == 0
    // __shared__ will be removed by educc-ast for the second pass.
    #define __shared__
#endif

#define EDU_CUDA_WARP_SIZE 32

namespace edu {
    namespace cuda {
        using namespace std;

        enum cudaError_t {
            cudaSuccess,
            cudaErrorInvalidDevicePointer,
            Not_Enough_Memory,
            Invalid_Device,
            Invalid_Grid_Dim,
            Invalid_Block_Dim
        };
        edu_thread_local cudaError_t last_error = cudaSuccess;
#define ret_err(err) {last_error = err; return err;}

        const char *cudaGetErrorString(cudaError_t err) {
            switch(err) {
            case cudaSuccess:
                return "Success";
            case cudaErrorInvalidDevicePointer:
                return "Not a valid device pointer";
            case Not_Enough_Memory:
                return "Out of memory";
            case Invalid_Device:
                return "Invalid device number";
            case Invalid_Grid_Dim:
                return "Illegal gridDim";
            case Invalid_Block_Dim:
                return "Illegal blockDim";
            default:
                abort();
            }
        }

        typedef unsigned int uint;

        template<typename T>
        struct vec3 {
            T x, y, z;

            vec3() : vec3(0,0,0) {}

            vec3(T x_, T y_ = 1, T z_ = 1) : x(x_), y(y_), z(z_) {
            }
        };

        typedef vec3<uint> uint3;
        typedef vec3<uint> dim3;

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
        int4 make_int4(int x, int y, int z, int w) { return {x,y,z,w}; }

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
            prop->warpSize = EDU_CUDA_WARP_SIZE;
            prop->maxThreadsPerMultiProcessor = 2048;
            prop->multiProcessorCount = 8;

            ret_err(cudaSuccess);
        }

        cudaError_t check_kernel_config(dim3 gridDim, dim3 blockDim) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

#define __dim_invalid(DIM, MAXSZ) (DIM.x > (uint)prop.MAXSZ[0])     \
                || (DIM.y > (uint)prop.MAXSZ[1])                    \
                || (DIM.z > (uint)prop.MAXSZ[2])

            if(__dim_invalid(gridDim, maxGridSize)) {
                ret_err(Invalid_Grid_Dim);
            }
            if(__dim_invalid(blockDim, maxThreadsDim)) {
                ret_err(Invalid_Block_Dim);
            }

#undef __dim_invalid

            if( (blockDim.x * blockDim.y * blockDim.z) > (uint)prop.maxThreadsPerBlock ) {
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

        cudaError_t cudaMemcpyToSymbol(const void *symbol,
                                       const void *src,
                                       size_t count,
                                       size_t offset = 0,
                                       cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
            mem::copy(memcpy_space[kind].dst, (void *)((char*)symbol + offset),
                      memcpy_space[kind].src, src,
                      count);
            ret_err(cudaSuccess);
        }

        cudaError_t cudaMemset(void *devPtr,
                               int value,
                               size_t count) {
            mem::set(mem::MemorySpace_Device, devPtr, value, count);
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
    }
}
