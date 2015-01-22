#define EDU_CUDA_ERR_THROW
#include <eduguard.h>

using namespace std;
using namespace edu;
using namespace edu::guard;

#define expect_fail(stmt) try {stmt; cerr << __FILE__ << ":" << __LINE__ << ": Expected failure" << endl; abort();} catch(...){}

//---
//--- 1D Arrays
//---
void test_array() {
    const int N = 20;
    float x[N];
    array1_guard_t<float, N> gx;

    assert(sizeof(x) == sizeof(gx));

    for(int i = 0; i < N; i++) {
        x[i] = i;
        gx[i] = i;
    }
    for(int i = 0; i < N; i++) {
        assert(x[i] == gx[i]);
    }

    for(int i = 0; i < N; i++) {
        x[i] *= 2;
        gx[i] *= 2;
    }
    for(int i = 0; i < N; i++) {
        assert(x[i] == gx[i]);
    }

    for(int i = 0; i < N; i++) {
        x[i] *= 10;
    }
    memcpy(gx, x, sizeof(gx));
    for(int i = 0; i < N; i++) {
        assert(x[i] == gx[i]);
    }

    for(int i = 0; i < N; i++) {
        assert( *(gx + i) == *(x + i) );
    }

    expect_fail(gx[-1]);
    expect_fail(gx[N]);
    expect_fail(gx - 1);
    expect_fail(gx + N);
}

//---
//--- 2D Arrays
//---
void test_array2() {
    const int M = 30;
    const int N = 20;
    float x[M][N];
    array2_guard_t<float, M, N> gx;

    assert(sizeof(x) == sizeof(gx));

#define __foreach(stmt)                         \
    for(int i = 0; i < M; i++) {                \
        for(int j = 0; j < N; j++) {            \
            stmt;                               \
        }                                       \
    }
#define __cmp() __foreach(assert(x[i][j] == gx[i][j]));


    __foreach(x[i][j] = 1 + i*j; gx[i][j] = 1 + i*j);
    __cmp();

    __foreach(x[i][j] *= 2; gx[i][j] *= 2);
    __cmp();

    __foreach(x[i][j] /= 3);
    __foreach(gx[i][j] = x[i][j]);
    __cmp();

    __foreach(gx[i][j] /= 3);
    __foreach(x[i][j] = gx[i][j]);
    __cmp();

    __foreach(gx[i][j] += 100);
    memcpy(x, gx, sizeof(gx));
    __cmp();

#undef __foreach
#undef __cmp

    expect_fail(gx[-1]);
    expect_fail(gx[M]);
    expect_fail(gx[0][-1]);
    expect_fail(gx[0][N]);
}

//---
//--- 3D Arrays
//---
void test_array3() {
    const int M = 30;
    const int N = 20;
    const int O = 10;
    float x[M][N][O];
    array3_guard_t<float, M, N, O> gx;

    assert(sizeof(x) == sizeof(gx));

#define __foreach(stmt)                         \
    for(int i = 0; i < M; i++) {                \
        for(int j = 0; j < N; j++) {            \
            for(int k = 0; k < O; k++) {        \
                stmt;                           \
            }                                   \
        }                                       \
    }
#define __cmp() __foreach(assert(x[i][j][k] == gx[i][j][k]));


    __foreach(x[i][j][k] = 1 + i*j*k; gx[i][j][k] = 1 + i*j*k);
    __cmp();

    __foreach(x[i][j][k] *= 2; gx[i][j][k] *= 2);
    __cmp();

    __foreach(x[i][j][k] /= 3);
    __foreach(gx[i][j][k] = x[i][j][k]);
    __cmp();

    __foreach(gx[i][j][k] /= 3);
    __foreach(x[i][j][k] = gx[i][j][k]);
    __cmp();

    __foreach(gx[i][j][k] += 100);
    memcpy(x, gx, sizeof(gx));
    __cmp();

#undef __foreach
#undef __cmp

    expect_fail(gx[-1]);
    expect_fail(gx[M]);
    expect_fail(gx[0][-1]);
    expect_fail(gx[0][N]);
    expect_fail(gx[0][0][-1]);
    expect_fail(gx[0][0][N]);
}

//---
//--- Pointers
//---
void test_ptr() {
    const size_t N = 100;
    ptr_guard_t<float> gp = (float*)mem::alloc(mem::MemorySpace_Host, N * sizeof(float));
    float *p = new float[100];

#define __foreach(stmt) for(size_t i = 0; i < N; i++) {stmt;}
#define __cmp() __foreach(assert(p[i] == gp[i]));

    __foreach(p[i] = i+1; gp[i] = i+1);
    __cmp();

    float *tp = p;
    ptr_guard_t<float> tgp = gp;

    __foreach(assert(*tp++ == *tgp++));

    tgp = gp + N - 1;
    expect_fail(*++tgp);

    tp = p - 1;
    tgp = gp - 1;
    __foreach(assert(*++tp == *++tgp));

    tp = p;
    tgp = gp;
    tp = p + 2;
    tgp = tgp + 2;
    assert(*tp == *tgp);
    tp += 2;
    tgp += 2;
    assert(*tp == *tgp);
    tp -= 2;
    tgp -= 2;
    assert(*tp == *tgp);

    expect_fail(*(gp + N + 1));
    expect_fail(*(gp - 1));

#undef __foreach
#undef __cmp
}

void __malloc(void ** p, unsigned len) {
    *p = mem::alloc(mem::MemorySpace_Host, len);
}

void test_cudaMalloc() {
    const unsigned N = 10;

    ptr_guard_t<float> p;
    __malloc((void**)&p, sizeof(float)*N);

    for(size_t i = 0; i < N; i++)
        p[i] = i;

    expect_fail(p[-1]);
    expect_fail(p[N]);
}

void test_write_callback() {
    clear_write_callback();
    clear_write_callback();

    function<void()> noop = [](){};

    set_write_callback(noop);
    clear_write_callback();

    set_write_callback(noop);
    clear_write_callback();
    clear_write_callback();

    set_write_callback(noop);
    expect_fail(set_write_callback(noop));
    clear_write_callback();
        
    {
        int ncallback = 0;
        function<void()> inc = [&ncallback]() {ncallback++;};
        set_write_callback(inc);

        const unsigned N = 10;
        array1_guard_t<unsigned, N> x;
        for(unsigned i = 0; i < N; i++) {
            x[i] = i;
        }
        assert(ncallback == N);

        ncallback = 0;
        for(unsigned i = 0; i < N; i++) {
            assert(x[i] == i);
        }
        assert(ncallback == 0);

        clear_write_callback();
    }

    {
        int ncallback = 0;
        function<void()> inc = [&ncallback]() {ncallback++;};
        set_write_callback(inc);

        const unsigned N = 10;
        ptr_guard_t<unsigned> x = (unsigned *)mem::alloc(mem::MemorySpace_Host, N * sizeof(unsigned));
        for(unsigned i = 0; i < N; i++) {
            x[i] = i;
        }
        assert(ncallback == N);

        ncallback = 0;
        for(unsigned i = 0; i < N; i++) {
            assert(x[i] == i);
        }
        assert(ncallback == 0);

        clear_write_callback();

        mem::dealloc(mem::MemorySpace_Host, x);
    }
}

int main(int argc, const char **argv) {
    test_array();
    test_array2();
    test_array3();

    test_ptr();
    test_cudaMalloc();

    test_write_callback();

    cout << "--- Passed guard tests ---" << endl;

    return 0;
}
