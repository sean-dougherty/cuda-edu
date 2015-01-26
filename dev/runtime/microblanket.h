#include <cstdlib>
#include <functional>
#include <setjmp.h>
#include <vector>

#ifdef __clang__
    // {get,set,make}context() are deprecated because the argument passing
    // mechanism violates standards. However, we don't pass any arguments.
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#if !defined(_WIN32)
    //
    // POSIX INCLUDES
    //
    #include <ucontext.h>
#else
    //
    // WINDOWS INCLUDES
    //
    #include <windows.h>
#endif
    
namespace microblanket {
    typedef int fiber_yield_t;
    const fiber_yield_t Fiber_Yield_Exit = -1;

    typedef unsigned fid_t;

    constexpr bool is_power_of_2(size_t x) {
        return x == 0 ? false : ((x & 1) ? (x == 1) : is_power_of_2(x>>1));
    }

    //------------------------------------------------------------
    //---
    //--- CLASS blanket_t
    //---
    //------------------------------------------------------------
    template<typename Fiber>
    class blanket_t {
    public:
        typedef std::function<void(Fiber *)> fiber_run_t;
          
    private:
        typedef typename Fiber::stack_buf_t stack_buf_t;
        // Contains stack buffer, pointer to fiber,  and fiber run function.
        // Assumes that stacks grow from high address to low address, keeps
        // fiber run function at the low bytes of the stack buffer.
        struct fiber_container_t {
            struct header_t {
                Fiber *fiber;
                fiber_run_t run_func;
            };
            stack_buf_t stack;
            void set_fiber(Fiber *f) {((header_t*)this)->fiber = f;}
            Fiber *fiber() {return ((header_t*)this)->fiber;}
            fiber_run_t *run_func() {return &((header_t*)this)->run_func;}
        };

        void *buf;
        fiber_container_t *containers;
        std::vector<Fiber> fibers;

        static_assert( sizeof(stack_buf_t) > sizeof(fiber_run_t),
                       "stack too small" );
        static_assert( is_power_of_2(sizeof(stack_buf_t)),
                       "stack size must be power of 2." );
    public:
        typedef typename std::vector<Fiber>::iterator iterator;
;
        iterator begin() {
            return fibers.begin();
        }
        iterator end() {
            return fibers.end();
        }

        // Transfers ownership of fibers.
        blanket_t(blanket_t &&other) {
            buf = other.buf;
            containers = other.containers;
            fibers = std::move(other.fibers);

            other.buf = nullptr;
            other.containers = nullptr;
        }

        // Allocates memory for fibers and their stacks.
        blanket_t(unsigned capacity) {
            // Custom align malloc logic so no worries about platform stuff.
            {
                const size_t sizeof_cont = sizeof(fiber_container_t);

                // Allocate enough for n+1 buffers.
                buf = malloc((capacity+1) * sizeof_cont);

                // Point to first container boundary.
                size_t cont_addr = (size_t(buf) + sizeof_cont) & ~(sizeof_cont-1);
                containers = (fiber_container_t*)cont_addr;
            }
            fibers.reserve(capacity);
        }

    public:
        // Destructs all the fibers.
        ~blanket_t() {
            // buf is null if we were moved.
            if(buf) {
                clear();
                free(buf);
                buf = nullptr;
                containers = nullptr;
            }
        }

        void clear() {
            for(unsigned i = 0; i < fibers.size(); i++) {
                fiber_container_t *fc = containers + i;
                fc->run_func()->~fiber_run_t();
            }
            fibers.resize(0);
        }

        // Creates fiber context, switches to it, executes run, and returns once
        // fiber yields or exits.
        fiber_yield_t spawn(fiber_run_t run) {
            unsigned i = fibers.size();
            assert(i < fibers.capacity());

            fibers.resize(i+1);
            fibers.back().fid = i;
            
            fiber_container_t *fc = containers + i;
            fc->set_fiber(&fibers.back());
            new (fc->run_func()) fiber_run_t(run);

            return create_fiber_context(fc);
        }

        Fiber &operator[](unsigned i) {
            return fibers[i];
        }

        // Returns the currently executing fiber. If no fiber is executing, will return garbage.
        static Fiber *current_fiber() {
            return current_container()->fiber();
        }
 
    private:
        blanket_t(const blanket_t &);

        static fiber_container_t *current_container() {
            int stack_var;
            size_t thiz = size_t(&stack_var) & ~(sizeof(fiber_container_t) - 1);
            return (fiber_container_t*)thiz;
        }

        // Entry-point for newly spawned fiber. We are now in the fiber's stack.
        static void fiber_entry() {
            fiber_container_t *fc = current_container();
            Fiber *f = fc->fiber();
            fiber_run_t &run_func = *fc->run_func();

            run_func(f);

            jmp_buf jmp_this;
            f->jmp_this = &jmp_this;
            if(0 == setjmp(jmp_this)) {
                longjmp(*f->jmp_main, Fiber_Yield_Exit);
            }
            abort(); // fiber has exited, but we've been asked to resume!
        }

#if !defined(_WIN32)
        //
        // POSIX create context
        // 
        fiber_yield_t create_fiber_context(fiber_container_t *fc) {
            ucontext_t ctxt;
            jmp_buf jmp_main;
        
            getcontext(&ctxt);
            const size_t Min_Context_Stack = 32 * 1024;
            if(sizeof(fc->stack) < Min_Context_Stack) {
                // OS will enforce a minimum size, but we can trick it.
                // Linux allows 4k, but Mac requires 32k.
                size_t padding = Min_Context_Stack - sizeof(fc->stack);
                ctxt.uc_stack.ss_sp = fc->stack - padding;
                ctxt.uc_stack.ss_size = sizeof(fc->stack) + padding;
            } else {
                ctxt.uc_stack.ss_sp = fc->stack;
                ctxt.uc_stack.ss_size = sizeof(fc->stack);
            }
            makecontext(&ctxt, fiber_entry, 0);

            fiber_yield_t rc = setjmp(jmp_main);
            if(rc == 0) {
                fc->fiber()->jmp_main = &jmp_main;
                setcontext(&ctxt);
            }
            return rc;
        }
#else
        //
        // WINDOWS create context
        // 
        fiber_yield_t create_fiber_context(fiber_container_t *fc) {
            CONTEXT ctxt;
            jmp_buf jmp_main;
        
            ctxt.ContextFlags = CONTEXT_FULL;
            if(!GetThreadContext(GetCurrentThread(), &ctxt)) {
                abort();
            }

            char *sp = (char*)fc->stack + sizeof(fc->stack);
    #if defined(_X86_)
            ctxt.Eip = (size_t) fiber_entry;
            ctxt.Esp = (size_t) (sp - 4);
    #else
            ctxt.Rip = (size_t) fiber_entry;
            ctxt.Rsp = (size_t) (sp - 40);
    #endif

            fiber_yield_t rc = setjmp(jmp_main);
            if(rc == 0) {
                fc->fiber()->jmp_main = &jmp_main;
                if(!SetThreadContext(GetCurrentThread(), &ctxt)) {
                    abort();
                }
            }
            return rc;
        }
#endif
    };

    //------------------------------------------------------------
    //---
    //--- CLASS fiber_t
    //---
    //------------------------------------------------------------
    template<size_t StackSize>
    class fiber_t {
        jmp_buf *jmp_this, *jmp_main;

        template<typename T>
            friend class blanket_t;

        typedef char stack_buf_t[StackSize];

    public:
        fid_t fid;

        static void yield(fiber_yield_t yield_value = 1) {
            assert(yield_value != 0);

            jmp_buf jmp_this;
            fiber_t<StackSize> *f = blanket_t<fiber_t<StackSize>>::current_fiber();
            f->jmp_this = &jmp_this;
            if(0 == setjmp(jmp_this)) {
                longjmp(*f->jmp_main, yield_value);
            }
        }

        fiber_yield_t resume() {
            jmp_buf jmp_main;
            fiber_yield_t rc = setjmp(jmp_main);
            if(rc == 0) {
                this->jmp_main = &jmp_main;
                longjmp(*jmp_this, 1);
            }
            return rc;
        }
    };
}

#ifdef __clang__
    #pragma clang diagnostic pop
#endif
