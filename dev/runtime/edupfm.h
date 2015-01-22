#pragma once

#ifdef __APPLE__
    // The ucontext API is deprecated because of its non-conformant
    // user entry prototype. The standard says "use threads!" but I
    // really need fibers. I use the __FIBER_SIGNATURE macros as a
    // sanity check to protect us when using this deprecated feature.
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    #define _XOPEN_SOURCE
#endif


#include <assert.h>
#include <cstdlib>
#include <ucontext.h>

namespace edu {
    namespace pfm {

//------------------------------------------------------------
//--- Threads
//------------------------------------------------------------
#ifdef __linux__
    #define edu_thread_local thread_local
#endif

#ifdef __APPLE__
    #define edu_thread_local __thread
#endif

//------------------------------------------------------------
//--- Atomics
//------------------------------------------------------------

#define atomicAdd(ptr, val) __sync_fetch_and_add(ptr, val)

//------------------------------------------------------------
//--- Fibers
//------------------------------------------------------------

#ifdef __APPLE__
      const size_t Min_Stack_Size = 32 * 1024;
#else
      const size_t Min_Stack_Size = 4 * 1024;
#endif

        // We pass these signatures as a sanity check.
        const size_t __FIBER_SIGNATURE_PREFIX = 0xdeadbeef;
        const size_t __FIBER_SIGNATURE_SUFFIX = 0xabcd1234;

        typedef ucontext_t fiber_context_t;
        typedef void (*fiber_context_entry_func_t)(void *user_data);

        // Entry-point for fiber. Calls user's fiber entry point.
        void __fiber_entry(size_t signature_prefix,
                           fiber_context_entry_func_t entry_func,
                           void *user_data,
                           size_t signature_suffix) {
            assert(signature_prefix == __FIBER_SIGNATURE_PREFIX);
            assert(signature_suffix == __FIBER_SIGNATURE_SUFFIX);

            entry_func(user_data);
        }

        // Initialize a fiber context.
        bool init_fiber_context(fiber_context_t *ctx,
                                fiber_context_t *ctx_exit,
                                void *stack,
                                size_t stacklen,
                                fiber_context_entry_func_t entry_func,
                                void *user_data) {
	    int rc = getcontext(ctx);
	    if(rc != 0)
                return false;
            ctx->uc_stack.ss_sp = stack;
            ctx->uc_stack.ss_size = stacklen;
            ctx->uc_link = ctx_exit;
            makecontext(ctx, (void (*)())__fiber_entry,
                        4, __FIBER_SIGNATURE_PREFIX,
                        entry_func,
                        user_data,
                        __FIBER_SIGNATURE_SUFFIX);
            return true;
        }

        // Switch into the fiber context specified in ctx_enter
        // and save the current state in ctx_save.
        bool switch_fiber_context(fiber_context_t *ctx_save,
                                  fiber_context_t *ctx_enter) {
	    int rc = swapcontext(ctx_save, ctx_enter);
	    return rc == 0;
        }

        // Dispose any resources for a context, if needed.
        void dispose_fiber_context(fiber_context_t &ctx) {
            //no-op
        }


    }
}

#ifdef __APPLE__
    #pragma clang diagnostic pop
#endif
