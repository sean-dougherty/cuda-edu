#pragma once

#include <edupfm.h>
#include <eduutil.h>

#include <cstring>
#include <iostream>
#include <map>

namespace edu {
    namespace mem {
        using namespace std;

        enum MemorySpace {MemorySpace_Host, MemorySpace_Device, MemorySpace_Unknown};

        ostream &operator<<(ostream &out, MemorySpace space) {
            switch(space) {
            case MemorySpace_Host: return out << "Host";
            case MemorySpace_Device: return out << "Device";
            case MemorySpace_Unknown: return out << "Unknown";
            default: return out << "INVALID";
            }
        }

        MemorySpace curr_space = MemorySpace_Host;

        struct AddressRange {
            size_t start;
            size_t end; //exclusive

            AddressRange(const void *ptr, size_t len) {
                start = size_t(ptr);
                end = start + len;
            }

            friend bool operator<(const AddressRange &a, const AddressRange &b) {
                return a.end <= b.start;
            }
        };

        struct Buffer {
            void *addr;
            size_t len;
            MemorySpace space;
            bool alloced;

            static Buffer alloc(MemorySpace space, unsigned len) {
                return {malloc(len), len, space, true};
            }

            // register
            static Buffer reg(MemorySpace space, void *addr, size_t len) {
                return {addr, len, space, false};
            }

            static Buffer get_universe() {
                return {nullptr, ~size_t(0), MemorySpace_Unknown, false};
            }

            static Buffer get_uninitialized() {
                return {nullptr, 0, MemorySpace_Unknown, false};
            }

            void dealloc() {
                if(!alloced) {
                    edu_err("Cannot free memory that wasn't malloc'd");
                }
                free(addr);
            }

            bool is_valid_space() {
                return (space == curr_space)
                    || (space == MemorySpace_Unknown);
            }

            bool is_valid(const void *addr, unsigned len, bool check_space = true) {
                return (!check_space || is_valid_space())
                    && (addr >= this->addr)
                    && ((const char*)addr + len) <= ((const char*)this->addr + this->len);
            }

            bool is_valid_offset(void *ptr, signed offset) {
                return is_valid_space()
                    && ((char*)ptr + offset >= (char*)addr)
                    && ((char*)ptr + offset < (char*)addr + len);
            }
        };

        typedef map<AddressRange, Buffer> BufferMap;
        BufferMap buffers;

        void warn_new() {
            edu_warn("Unknown memory location, did you use new or shared_ptr/unique_ptr? Please use malloc() or cudaMallocHost() so that errors can be more easily detected by cuda-edu.");
        }

        bool find_buf(const void *addr, Buffer *buf, size_t len = 1) {
            auto it = buffers.find({addr,len});
            if(it != buffers.end()) {
                *buf = it->second;
                return true;
            }
            return false;
        }

        void set_space(MemorySpace space) {
            curr_space = space;
        }

        void register_memory(MemorySpace space, void *addr, size_t len) {
            Buffer buf;
            if(find_buf(addr, &buf, len)) {
                edu_panic("Memory already registered!");
            }
            buf = Buffer::reg(space, addr, len);
            buffers[{addr, len}] = buf;
        }

        void *alloc(MemorySpace space, unsigned len) {
            Buffer buf = Buffer::alloc(space, len);
            buffers[{buf.addr,buf.len}] = buf;
            return buf.addr;
        }

        void dealloc(MemorySpace space, void *addr) {
            Buffer buf;
            if(!find_buf(addr, &buf)) {
                edu_err("Invalid buffer.");
            }
            if(buf.addr != addr) {
                edu_err("Attempting to free memory in middle of allocated buffer.");
            }
            if(space != buf.space) {
                edu_err("Requested to free memory in " << space << ", but provided address in " << buf.space);
            }
            buffers.erase({buf.addr, buf.len});
            buf.dealloc();
        }

        void copy(MemorySpace dst_space, void *dst,
                  MemorySpace src_space, const void *src,
                  unsigned len) {

            Buffer dst_buf;
            Buffer src_buf;

#define __check(BUF, SPACE, PTR, DIR) {                                 \
                if(!find_buf(PTR, &BUF)) {                              \
                    if(SPACE == MemorySpace_Host) {                     \
                        warn_new();                                     \
                    } else {                                            \
                        edu_err("Invalid Device buffer specified.");    \
                    }                                                   \
                } else {                                                \
                    if(BUF.space != SPACE) {                            \
                        edu_err("Attempting to copy " << DIR << " " << SPACE \
                                << " but provided address in " << BUF.space); \
                    }                                                   \
                    if(!BUF.is_valid(PTR, len, false)) {                \
                        edu_err("Invalid '" << DIR << "' address or bounds."); \
                    }                                                   \
                }                                                       \
            }

            __check(dst_buf, dst_space, dst, "to");
            __check(src_buf, src_space, src, "from");

            memcpy(dst, src, len);

#undef __check
        }

        void set(MemorySpace ptr_space, void *ptr, int value, size_t count) {
            Buffer buf;
            if(!find_buf(ptr, &buf)) {
                //todo: refactor to share common logic with copy()
                if(ptr_space == MemorySpace_Host) {
                    warn_new();
                } else {
                    edu_err("Invalid Device buffer specified.");
                }
            } else {
                if(buf.space != ptr_space) {
                    edu_err("Requesting to set buffer in " << ptr_space
                            << " but provided address in " << buf.space);
                }
                if(!buf.is_valid(ptr, count, false)) {
                    edu_err("Invalid address bounds.");
                }
            }

            memset(ptr, value, count);
        }
    }
}
