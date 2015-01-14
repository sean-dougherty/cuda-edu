#pragma once

#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

namespace edu {
    namespace wb {
        using namespace std;

#define wbTime_start(x...)
#define wbTime_stop(x...)

#define edu_err(x...) {cerr << x << endl; exit(1);}

        struct wbArg_t {
            int argc;
            char **argv;

            string data_dir() {
                if(argc != 2) {
                    cerr << "usage: " << argv[0] << " data/i" << endl;
                    cerr << endl;
                    cerr << "example: " << argv[0] << " data/2" << endl;
                    exit(1);
                }
                return argv[1];
            }
            string get_input_path(int index) {
                stringstream ss;
                ss << data_dir() << "/input" << index << ".raw";
                return ss.str();
            }
            string get_output_path() {
                stringstream ss;
                ss << data_dir() << "/output.raw";
                return ss.str();
            }
        };

        typedef std::string wbFile_t;

        float *read_data(string path, int *length) {
            ifstream in(path.c_str());
            if(!in) {
                edu_err("Failed opening input: " << path);
            }
            in >> *length;
            float *buf = (float *)malloc(*length * sizeof(float));
            for(int i = 0; i < *length; i++) {
                in >> buf[i];
                if(!in) {
                    edu_err("Failed reading input: " << path);
                }
            }
            return buf;
        }

        float *read_data(string path, int *rows, int *cols) {
            ifstream in(path.c_str());
            if(!in) {
                edu_err("Failed opening input: " << path);
            }
            in >> *rows;
            in >> *cols;
            float *buf = (float *)malloc(*rows * *cols * sizeof(float));
            for(int i = 0; i < *rows * *cols; i++) {
                in >> buf[i];
                if(!in) {
                    edu_err("Failed reading input: " << path);
                }
            }
            return buf;
        }

        wbArg_t wbArg_read(int argc, char **argv) {
            return {argc, argv};
        }

        wbFile_t wbArg_getInputFile(wbArg_t args, int index) {
            return args.get_input_path(index);
        }

        void *wbImport(wbFile_t f, int *length) {
            return read_data(f, length);
        }

        void *wbImport(wbFile_t f, int *rows, int *cols) {
            return read_data(f, rows, cols);
        }

        void wbSolution(wbArg_t args, void *output, int length) {
            unique_ptr<float> expected(read_data(args.get_output_path(), &length));
            float *actual = (float *)output;

            for(int i = 0; i < length; i++) {
                float e = expected.get()[i];
                float a = actual[i];
                if( fabs(a - e) > (1e-3 * e) ) {
                    edu_err("Results mismatch at index " << i << ". Expected " << e << ", found " << a << ".");
                }
            }

            cout << "Solution correct." << endl;
        }

        void wbSolution(wbArg_t args, void *output, int rows_, int cols_) {
            int rows, cols;
            unique_ptr<float> expected(read_data(args.get_output_path(), &rows, &cols));
            if(rows != rows_) {
                edu_err("Incorrect number of rows. Expected " << rows << ", found " << rows_);
            }
            if(cols != cols_) {
                edu_err("Incorrect number of cols. Expected " << cols << ", found " << cols_);
            }

            float *actual = (float *)output;

            for(int i = 0; i < (rows * cols); i++) {
                float e = expected.get()[i];
                float a = actual[i];
                if( fabs(a - e) > (1e-3 * e) ) {
                    edu_err("Results mismatch at index " << i << ". Expected " << e << ", found " << a << ".");
                }
            }

            cout << "Solution correct." << endl;
        }

        enum wbLog_level_t {
            ERROR,
            TRACE
        };
        template<typename T>
            void __wbLog(T x) {
            cout << x;
        }
        template<typename T, typename... U>
            void __wbLog(T arg0, U... args) {
            cout << arg0;
            __wbLog(args...);
        }
        template<typename... T>
            void wbLog(wbLog_level_t lvl, T... args) {
            __wbLog(args...);
            cout << endl;
        }
    }

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
            void *result = malloc(length);
            if(!result)
                ret_err(Not_Enough_Memory);
            *ptr = result;
            ret_err(cudaSuccess);
        }

        cudaError_t cudaFree(void *ptr) {
            free(ptr);
            ret_err(cudaSuccess);
        }

        cudaError_t cudaMallocHost(void **ptr, size_t length) {
            void *result = malloc(length);
            if(!result)
                ret_err(Not_Enough_Memory);
            *ptr = result;
            ret_err(cudaSuccess);
        }

        cudaError_t cudaFreeHost(void *ptr) {
            free(ptr);
            ret_err(cudaSuccess);
        }

        enum cudaMemcpyKind {
            cudaMemcpyHostToHost,
            cudaMemcpyHostToDevice,
            cudaMemcpyDeviceToHost,
            cudaMemcpyDeviceToDevice
        };

        cudaError_t cudaMemcpy(void *dst,
                               const void *src,
                               size_t count,
                               cudaMemcpyKind kind) {
            memcpy(dst, src, count);
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
                
            }
        };

#define __syncthreads() {                        \
            static Barrier barrier_##__LINE__; barrier_##__LINE__.sync(); \
        }
#endif

#define atomicAdd(ptr, val) __sync_fetch_and_add(ptr, val)
    }
}

#define __global__
#define __device__
#define __host__

#define __shared__ static

using namespace edu::wb;
using namespace edu::cuda;
