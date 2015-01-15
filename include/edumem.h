#pragma once

#include <cstring>
#include <iostream>
#include <map>
#include <sys/mman.h>
#include <eduutil.h>

namespace edu {
    namespace mem {
        using namespace std;

        enum MemorySpace {
            MemorySpace_Host = 0, MemorySpace_Device = 1, __MemorySpace_N = 2
        };
        ostream &operator<<(ostream &out, MemorySpace space) {
            switch(space) {
            case MemorySpace_Host: return out << "Host";
            case MemorySpace_Device: return out << "Device";
            default: return out << "INVALID";
            }
        }

        MemorySpace curr_space = MemorySpace_Host;

        struct Buffer {
            void *addr;
            unsigned len;
            MemorySpace space;

            static Buffer alloc(MemorySpace space, unsigned len) {
                Buffer buf;
                buf.addr = mmap(nullptr,
                                len,
                                PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS,
                                0, 0);
                edu_errif(buf.addr == nullptr);
                buf.len = len;
                buf.space = space;
                return buf;
            }

            void dealloc() {
                edu_errif(0 != munmap(addr, len));
            }

            void activate() {
                edu_errif(0 != mprotect(addr, len, PROT_READ | PROT_WRITE));
            }

            void deactivate() {
                edu_errif(0 != mprotect(addr, len, PROT_NONE));
            }

            bool valid(const void *addr, unsigned len) {
                return (addr >= this->addr)
                    && ((const char*)addr + len) <= ((const char*)this->addr + this->len);
            }
        };

        typedef map<const void *, Buffer> BufferMap;
        BufferMap spaces[__MemorySpace_N];

        bool find_buf(const void *addr, Buffer *buf, BufferMap **bufmap = nullptr) {
#define __tryget(space) {                                               \
                auto it = spaces[space].find(addr);                     \
                if(it != spaces[space].end()) {                         \
                    if(bufmap) *bufmap = &spaces[space];                \
                    *buf = it->second;                                  \
                    return true;                                        \
                }                                                       \
            }

            __tryget(MemorySpace_Host);
            __tryget(MemorySpace_Device);
#undef __tryget
            return false;
        }

        void activate_space(MemorySpace space) {
            BufferMap &bufs = spaces[space];
            for(auto &kv: bufs) {
                Buffer &buf = kv.second;
                buf.activate();
            }
        }

        void deactivate_space(MemorySpace space) {
            BufferMap &bufs = spaces[space];
            for(auto &kv: bufs) {
                Buffer &buf = kv.second;
                buf.deactivate();
            }
        }

        void set_space(MemorySpace space) {
            activate_space(space);
            deactivate_space(MemorySpace((space + 1) % __MemorySpace_N));
            curr_space = space;
        }

        void *alloc(MemorySpace space, unsigned len) {
            Buffer buf = Buffer::alloc(space, len);
            if(space == curr_space) {
                buf.activate();
            } else {
                buf.deactivate();
            }
            spaces[space][buf.addr] = buf;
            return buf.addr;
        }

        void dealloc(void *addr) {
            Buffer buf;
            BufferMap *bufmap;
            edu_errif(!find_buf(addr, &buf, &bufmap));

            bufmap->erase(buf.addr);
            buf.dealloc();
        }

        void copy(MemorySpace dst_space, void *dst,
                  MemorySpace src_space, const void *src,
                  unsigned len) {

            Buffer dst_buf;
            Buffer src_buf;

#define __acquire(BUF, SPACE, PTR, DIR) {                               \
                edu_errif(!find_buf(PTR, &BUF));                        \
                if(BUF.space != SPACE) {                                \
                    edu_err("Attempting to copy " << DIR << " " << SPACE \
                            << " but provided address in " << BUF.space); \
                }                                                       \
                if(BUF.space != curr_space) {                           \
                    BUF.activate();                                     \
                }                                                       \
                edu_errif(!BUF.valid(PTR, len));                        \
            }

#define __release(BUF) {                        \
                if(BUF.space != curr_space) {   \
                    BUF.deactivate();           \
                }                               \
            }

            __acquire(dst_buf, dst_space, dst, "to");                      
            __acquire(src_buf, src_space, src, "from");

            memcpy(dst, src, len);

            __release(dst_buf);
            __release(src_buf);

#undef __acquire
#undef __release
        }
    }
}
