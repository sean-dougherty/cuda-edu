// Make sure educc correctly identifies memory properties of symbols.

// MP 1
#include <wb.h>
#include <assert.h>

using namespace std;

const size_t N = 10;

__device__ float device_global_float;
__device__ float device_global_float_array[N];
__constant__ float constant_float;
__constant__ float constant_float_array[N];

float implicit_host_float;
float implicit_host_float_array[N];
__host__ float host_float;
__host__ float host_float_array[N];

#define SZF (sizeof(float))
#define SZFA (sizeof(float) * N)

int main(int argc, char **argv) {

#if EDU_CUDA_COMPILE_PASS == 1
#define check(PTR, SPACE, LEN) {                                \
        edu::mem::Buffer buf;                                   \
        edu_assert(find_buf(PTR, &buf));                        \
        edu_assert(buf.addr == (PTR));                          \
        edu_assert(buf.space == edu::mem::MemorySpace_##SPACE); \
        edu_assert(buf.len == (LEN));                           \
    }
#else
    // Define enough to make the Pass 0 compile happy.
    #define Device 0
    #define Host 0 
    extern void check(...);
#endif

    check(&device_global_float, Device, SZF);
    check(&device_global_float_array, Device, SZFA);
    check(&constant_float, Device, SZF);
    check(&constant_float_array, Device, SZFA);

    check(&implicit_host_float, Host, SZF);
    check(&implicit_host_float_array, Host, SZFA);
    check(&host_float, Host, SZF);
    check(&host_float_array, Host, SZFA);

    float *device_malloc;
    cudaMalloc((void **)&device_malloc, SZFA);
    check(device_malloc, Device, SZFA);

    float *host_malloc = (float *)malloc(SZFA);
    check(host_malloc, Host, SZFA);

    printf("--- mpbuf tests passed ---\n");

    return 0;
}
