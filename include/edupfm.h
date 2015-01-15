#pragma once

#ifdef __linux__
#include <sys/mman.h>
#endif

#include <cstdlib>

namespace edu {
    namespace pfm {

        enum mem_access_t {
            MemAccess_None,
            MemAccess_ReadWrite
        };

#ifdef __linux__

#define atomicAdd(ptr, val) __sync_fetch_and_add(ptr, val)

        void *alloc(unsigned len) {
            return mmap(nullptr,
                        len,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1, // some systems require fd = -1 for anonymous
                        0);
        }

        // true = succeed
        bool dealloc(void *addr, unsigned len) {
            return 0 == munmap(addr, len);
        }

        // true = succeed
        bool set_mem_access(void *addr, unsigned len, mem_access_t access) {
            int prot;
            switch(access) {
            case MemAccess_None:
                prot = PROT_NONE;
                break;
            case MemAccess_ReadWrite:
                prot = PROT_READ | PROT_WRITE;
                break;
            default:
                abort();
            }

            return 0 == mprotect(addr, len, prot);
        }

#endif

    }
}
