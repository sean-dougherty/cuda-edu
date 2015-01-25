#define EDU_CUDA_ERR_THROW
#include <edumem.h>

#include <memory>
#include <thread>
#include <vector>

using namespace std;
using namespace edu;
using namespace edu::pfm;

#define expect_fail(stmt) try {stmt; cerr << __FILE__ << ":" << __LINE__ << ": Expected failure" << endl; abort();} catch(...){}

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

int main(int argc, const char **argv) {
    test_atomic();

    cout << "--- Passed pfm tests ---" << endl;

    return 0;
}
