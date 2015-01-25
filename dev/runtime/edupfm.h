#pragma once

#include <eduutil.h>

#include <assert.h>
#include <cstdlib>
#include <thread>
#include <ucontext.h>

namespace edu {
    namespace pfm {

//------------------------------------------------------------
//--- Threads
//------------------------------------------------------------
#ifdef __APPLE__
    // Apple's clang++ doesn't support thread_local yet
    #define edu_thread_local __thread
#else
    // Use the C++11 standard
    #define edu_thread_local thread_local
#endif

        unsigned get_thread_count() {
            const char *env = getenv("EDU_CUDA_THREAD_COUNT");
            if(env) {
                return util::parse_uint(env,
                                        "EDU_CUDA_THREAD_COUNT environment variable",
                                        1, 16);
            } else {
                unsigned n = std::thread::hardware_concurrency();
                if(n == 0) {
                    n = 2;
                    edu_warn("Failed determining hardware concurrency. Defaulting to " << n << " threads.");
                }
                return n;
            }
        }

//------------------------------------------------------------
//--- Atomics
//------------------------------------------------------------

#define atomicAdd(ptr, val) __sync_fetch_and_add(ptr, val)

    }
}
