// MP 1
#include <wb.h>
#include <assert.h>

using namespace std;

#define TILE_WIDTH 8

__global__ void convolute(float *in1, float *in2, float *out, int len) {
    __shared__ float tile[TILE_WIDTH];

    int tile_start = -1 + (TILE_WIDTH - 2)*blockIdx.x;
    int in_load_index = tile_start + threadIdx.x;

    if( (in_load_index == -1) || (in_load_index >= len) ) {
        tile[threadIdx.x] = 0.0;
    } else {
        tile[threadIdx.x] = in1[in_load_index];
    }

    __syncthreads();

    int out_store_index = tile_start + threadIdx.x + 1;
    if( (out_store_index < len) && (threadIdx.x < blockDim.x - 2) ) {
        float sum = tile[threadIdx.x] + tile[threadIdx.x+1] + tile[threadIdx.x + 2];
        out[tile_start + threadIdx.x + 1] = sum;
    }
}

void convolute_cpu(float *in1, float *in2, float *out, int len) {
    for(int i = 0; i < len; i++) {
        float sum = 0.0;
        if(i > 0)
            sum += in1[i - 1];
        if(i < (len-1))
            sum += in1[i + 1];
        sum += in1[i];

        out[i] = sum;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    float *hostSolution;
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = ( float * )malloc(inputLength * sizeof(float));
    hostSolution = ( float * )malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    convolute_cpu(hostInput1, hostInput2, hostSolution, inputLength);

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
    cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
    cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    int nblocks = (inputLength - 1) / (TILE_WIDTH - 2) + 1;
    int nthreads = TILE_WIDTH;
    cout << "nblocks = " << nblocks << endl;
    convolute<<<nblocks,nthreads>>> (deviceInput1,
                       deviceInput2, deviceOutput, inputLength );

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    for(int i = 0; i < inputLength; i++) {
        if( fabs(hostSolution[i] - hostOutput[i]) / hostSolution[i] > (1e-3 * hostSolution[i]) ) {
            cerr << "Mismatch at " << i << ", expected " << hostSolution[i] << ", found " << hostOutput[i] << endl;
            exit(1);
        }
    }

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(hostSolution);

    return 0;
}
