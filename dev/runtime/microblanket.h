#include <cstdlib>
#include <functional>

#ifdef __clang__
    // {get,set,make}context() are deprecated because the argument passing
    // mechanism violates standards. However, we don't pass any arguments.
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <setjmp.h>
#include <ucontext.h>
#ifdef __clang__
    #pragma clang diagnostic push
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
        // Contains stack buffer, fiber, and fiber run function. Assumes that stacks grow
        // from high address to low address, keeps fiber datastructure and its run
        // function at the low bytes of the stack buffer.
        struct fiber_container_t {
            stack_buf_t stack;
            Fiber *fiber() {return (Fiber*)this;}
            fiber_run_t *run_func() {return (fiber_run_t*)(fiber() + 1);}
        };

        void *buf;
        fiber_container_t *containers;
        unsigned capacity;
        unsigned nspawned;

        // Unless Fiber is gigantic, stack must be power of 2 >= 4096.
        static_assert( sizeof(stack_buf_t) > (sizeof(Fiber)+sizeof(fiber_run_t)+2048),
                       "stack too small" );
        static_assert( is_power_of_2(sizeof(stack_buf_t)),
                       "stack size must be power of 2." );
    public:
        struct iterator : public std::iterator<std::input_iterator_tag, Fiber> {
        private:
            fiber_container_t *ptr;
        public:
            iterator(fiber_container_t *ptr_) : ptr(ptr_) {
            }
            bool operator!=(const iterator &other) const {
                return ptr != other.ptr;
            }
            iterator &operator++() {
                ++ptr;
                return *this;
            }
            Fiber &operator*() {
                return *(Fiber*)ptr;
            }
        };
        iterator begin() {
            return iterator(containers);
        }
        iterator end() {
            return iterator(containers+nspawned);
        }

        // Transfers ownership of fibers.
        blanket_t(blanket_t &&other) {
            buf = other.buf;
            containers = other.containers;
            capacity = other.capacity;

            other.buf = nullptr;
            other.containers = nullptr;
            other.capacity = 0;
        }

        // Allocates fibers and their stacks. Fibers
        // not yet constructed.
        blanket_t(unsigned capacity_)
            : capacity(capacity_)
            , nspawned(0) {
            // Custom align malloc logic so no worries about platform stuff.
            {
                const size_t sizeof_cont = sizeof(fiber_container_t);

                // Allocate enough for n+1 buffers.
                buf = malloc((capacity+1) * sizeof_cont);

                // Point to first container boundary.
                size_t cont_addr = (size_t(buf) + sizeof_cont) & ~(sizeof_cont-1);
                containers = (fiber_container_t*)cont_addr;
            }
        }

    private:
        void construct_fiber(unsigned i) {
            fiber_container_t *fc = containers + i;
            Fiber *f = fc->fiber();
            new (f) Fiber();
            f->fid = i;
            fiber_run_t *run_func = fc->run_func();
            new (run_func) fiber_run_t();
        }

        void destruct_fiber(unsigned i) {
            fiber_container_t *fc = containers + i;
            fc->fiber()->~Fiber();
            fc->run_func()->~fiber_run_t();
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
            for(unsigned i = 0; i < nspawned; i++) {
                destruct_fiber(i);
            }
            nspawned = 0;
        }

        // Creates fiber context, switches to it, executes run, and returns once
        // fiber yields or exits.
        fiber_yield_t spawn(fiber_run_t run) {
            assert(nspawned < capacity);

            unsigned i = nspawned++;

            construct_fiber(i);
            fiber_container_t *fc = containers + i;
            *fc->run_func() = run;

            return create_fiber_context(fc);
        }

        Fiber &operator[](unsigned i) {
            return *containers[i].fiber();
        }

        // Get the fiber at index i.
        Fiber *get_fiber(unsigned i) {
            return containers[i].fiber();
        }

        // Returns the currently executing fiber. If no fiber is executing, will return garbage.
        static Fiber *current_fiber() {
            return current_container()->fiber();
        }
 
    private:
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

        static fiber_container_t *current_container() {
            int stack_var;
            size_t thiz = size_t(&stack_var) & ~(sizeof(fiber_container_t) - 1);
            return (fiber_container_t*)thiz;
        }

        static void fiber_entry() {
            fiber_container_t *fc = current_container();
            Fiber *f = fc->fiber();
            fiber_run_t &run_func = *fc->run_func();

            run_func(f);

            if(0 == setjmp(f->jmp_this)) {
                longjmp(*f->jmp_main, Fiber_Yield_Exit);
            }
            abort(); // fiber has exited, but we've been asked to resume!
        }

        blanket_t(const blanket_t &);
    };

    //------------------------------------------------------------
    //---
    //--- CLASS fiber_t
    //---
    //------------------------------------------------------------
    template<size_t StackSize>
    class fiber_t {
        jmp_buf jmp_this, *jmp_main;

        template<typename T>
            friend class blanket_t;

        typedef char stack_buf_t[StackSize];

    public:
        fid_t fid;

        static void yield(fiber_yield_t yield_value = 1) {
            assert(yield_value != 0);

            fiber_t<StackSize> *f = blanket_t<fiber_t<StackSize>>::current_fiber();
            if(0 == setjmp(f->jmp_this)) {
                longjmp(*f->jmp_main, yield_value);
            }
        }

        fiber_yield_t resume() {
            jmp_buf jmp_main;
            fiber_yield_t rc = setjmp(jmp_main);
            if(rc == 0) {
                this->jmp_main = &jmp_main;
                longjmp(jmp_this, 1);
            }
            return rc;
        }
    };
}
