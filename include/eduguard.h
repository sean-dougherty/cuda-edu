#pragma once

#include <edumem.h>

namespace edu {
    namespace guard {
        using namespace std;
        using edu::mem::Buffer;

        template<typename T>
            struct ptr_guard_t {
                T *ptr;
                Buffer buf;
                void *buf_ptr; // detect when changed via cudaMalloc(void **);

                ptr_guard_t() {
                    ptr = nullptr;
                    buf = Buffer::get_uninitialized();
                    buf_ptr = nullptr;
                }

                ptr_guard_t(T *ptr_) : ptr(ptr_) {
                    if(!mem::find_buf(ptr, &buf)) {
                        buf = Buffer::get_universe();
                    }
                    buf_ptr = ptr;
                }
                ptr_guard_t(T *ptr_, Buffer buf_) : ptr(ptr_), buf(buf_), buf_ptr(ptr_) {
                }

                void check_offset(int i) {
                    if(ptr != buf_ptr) { // must have been changed by cudaMalloc()
                        if(!mem::find_buf(ptr, &buf)) {
                            buf = Buffer::get_universe();
                            buf_ptr = ptr;
                        }
                    }
                    if(!buf.is_valid_offset(ptr, i * sizeof(T))) {
                        edu_err("Buffer bounds violated.");
                    }
                }

                T &operator[](int i) {
                    check_offset(i);
                    return ptr[i];
                }

                T &operator*() {
                    return *ptr;
                }

                operator T*() {
                    return ptr;
                }

                ptr_guard_t operator+(int i) {
                    check_offset(i);
                    return ptr_guard_t(ptr + i, buf);
                }

                ptr_guard_t operator-(int i) {
                    return *this + (-i);
                }

                ptr_guard_t &operator+=(int i) {
                    check_offset(i);
                    ptr += i;
                    return *this;
                }

                ptr_guard_t &operator-=(int i) {
                    return *this += -i;
                }

                // prefix
                ptr_guard_t &operator++() {
                    return *this += 1;
                }
                // postfix
                ptr_guard_t operator++(int) {
                    ptr_guard_t result = *this;
                    *this += 1;
                    return result;
                }

                // prefix
                ptr_guard_t &operator--() {
                    return *this -= 1;
                }
                // postfix
                ptr_guard_t operator--(int) {
                    ptr_guard_t result = *this;
                    *this -= 1;
                    return result;
                }
            };


        template<typename T, int xlen>
            struct array_guard_t {
                T data[xlen];

                T &operator[](int i) {
                    edu_assert(i >= 0 && i < xlen);
                    return data[i]; 
                }

                operator T*() {
                    return data;
                }

                T *operator+(int i) {
                    edu_assert(i >= 0 && (i*sizeof(T)) < sizeof(data));
                    return data + i;
                }

                T *operator-(int i) {
                    return *this + (-i);
                }
            };

        template<typename T, int xlen, int ylen>
            struct array_guard2_t {
                array_guard_t<T, ylen> data[xlen];

                array_guard_t<T, ylen> &operator[](int i) {
                    edu_assert(i >= 0 && i < xlen);
                    return data[i]; 
                }

                operator T*() {
                    return (T*)data;
                }
            };

        template<typename T, int xlen, int ylen, int zlen>
            struct array_guard3_t {
                array_guard2_t<T, ylen, zlen> data[xlen];

                array_guard2_t<T, ylen, zlen> &operator[](int i) {
                    edu_assert(i >= 0 && i < xlen);
                    return data[i]; 
                }

                operator T*() {
                    return (T*)data;
                }
            };

    }
}
