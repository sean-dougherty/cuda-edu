#define EDU_CUDA_ERR_THROW
#include <edumem.h>

#include <memory>
#include <thread>
#include <vector>

using namespace std;
using namespace edu;
using namespace edu::pfm;

#define expect_fail(stmt) try {stmt; cerr << __FILE__ << ":" << __LINE__ << ": Expected failure" << endl; abort();} catch(...){}

void test_fiber() {
    struct fiber_t {
        int state = 0;
        fiber_context_t ctx_main, ctx_fiber;

        static void entry(void *user_data) {
            fiber_t *fiber = (fiber_t *)user_data;

            assert(0 == fiber->state);
            fiber->state++;
            assert(1 == fiber->state);
            assert(switch_fiber_context(&fiber->ctx_fiber, &fiber->ctx_main));
            assert(2 == fiber->state);
            fiber->state++;
        }
    } fiber;
    char stack[4096];

    assert(init_fiber_context(&fiber.ctx_fiber,
                              &fiber.ctx_main,
                              stack,
                              sizeof(stack),
                              fiber_t::entry,
                              &fiber));
    assert(0 == fiber.state);
    assert(switch_fiber_context(&fiber.ctx_main, &fiber.ctx_fiber));
    assert(1 == fiber.state);
    fiber.state++;
    assert(switch_fiber_context(&fiber.ctx_main, &fiber.ctx_fiber));
    assert(3 == fiber.state);

    dispose_fiber_context(fiber.ctx_main);
    dispose_fiber_context(fiber.ctx_fiber);
}

void test_atomic() {
    int accum = 100;

    assert(100 == atomicAdd(&accum, 10));
    assert(110 == accum);
    
    const unsigned int NTHREADS = 2;
    const unsigned int NITERATIONS = 100000000u; // 1e8
    const unsigned int CORRECT = NTHREADS * NITERATIONS;

    // Verify we can create incorrect answer.
    {
        cout << "Stress testing (invalid) atomic add..." << flush;

        accum = 0;
        vector<unique_ptr<thread>> threads;
        for(unsigned int i = 0; i < NTHREADS; i++) {
            threads.emplace_back(
                new thread(
                    [&accum]() {
                        for(unsigned int j = 0; j < NITERATIONS; j++) {
                            accum++;
                        }
                    }));
        }
        for(auto &t: threads)
            t->join();
        assert(accum != CORRECT);

        cout << "OK" << endl;
    }

    // Now test atomic
    {
        cout << "Stress testing (valid) atomic add..." << flush;

        accum = 0;
        vector<unique_ptr<thread>> threads;
        for(unsigned int i = 0; i < NTHREADS; i++) {
            threads.emplace_back(
                new thread(
                    [&accum]() {
                        for(unsigned int j = 0; j < NITERATIONS; j++) {
                            atomicAdd(&accum, 1);
                        }
                    }));
        }
        for(auto &t: threads)
            t->join();
        assert(accum == CORRECT);

        cout << "OK" << endl;
    }
}

void test_alloc() {
    const unsigned N = 10000;

    char *x = (char *)alloc(N);
    
    for(unsigned i = 0; i < N; i++)
        x[i] = i;

    assert(dealloc(x, N));
}

void test_mprotect() {
    const unsigned N = 10;

    char *x = (char *)alloc(N);
    x[0] = 1;
    assert(1 == x[0]);

    assert(set_mem_access(x, N, MemAccess_None));
    // Can't easily do a pfm-agnostic test...
    // Should verify sigsegv on access here.
    assert(set_mem_access(x, N, MemAccess_ReadWrite));

    assert(dealloc(x, N));
}

int main(int argc, const char **argv) {
    test_fiber();
    test_atomic();
    test_alloc();
    test_mprotect();

    cout << "--- Passed pfm tests ---" << endl;

    return 0;
}
