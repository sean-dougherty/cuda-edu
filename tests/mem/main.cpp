#define EDU_CUDA_ERR_THROW
#include <edumem.h>

using namespace std;
using namespace edu;
using namespace edu::mem;

#define expect_fail(stmt) try {stmt; cerr << __FILE__ << ":" << __LINE__ << ": Expected failure" << endl; abort();} catch(...){}

void test_find() {
    const size_t N = 10;
    float *p = (float *)mem::alloc(MemorySpace_Host, N * sizeof(float));
    float *p1 = (float *)mem::alloc(MemorySpace_Host, N * sizeof(float));
    float *p2 = (float *)mem::alloc(MemorySpace_Host, N * sizeof(float));
    assert(p);

    Buffer buf;

    assert(find_buf(p, &buf));
    assert(buf.addr == p);
    assert(buf.len == N*sizeof(float));

    assert(find_buf(p1, &buf));
    assert(buf.addr == p1);
    assert(buf.len == N*sizeof(float));

    assert(find_buf(p2, &buf));
    assert(buf.addr == p2);
    assert(buf.len == N*sizeof(float));

    assert(find_buf(p+1, &buf));
    assert(buf.addr == p);
    assert(buf.len == N*sizeof(float));

    assert(find_buf(p1+1, &buf));
    assert(buf.addr == p1);
    assert(buf.len == N*sizeof(float));

    assert(find_buf(p2+2, &buf));
    assert(buf.addr == p2);
    assert(buf.len == N*sizeof(float));

    assert(!find_buf(p-1, &buf));
    assert(!find_buf(p2+N, &buf));
}

int main(int argc, const char **argv) {
    test_find();

    cout << "--- Passed mem tests ---" << endl;

    return 0;
}
