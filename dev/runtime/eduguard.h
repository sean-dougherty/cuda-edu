#pragma once

#include <edumem.h>

#include <functional>

namespace edu {
    namespace guard {
        using namespace std;
        using edu::mem::Buffer;

        enum write_callback_state_t {
            Write_Callback_Clear,
            Write_Callback_Set
        } write_callback_state = Write_Callback_Clear;
        
        function<void()> write_callback = [](){};

        void set_write_callback(function<void()> write_callback_) {
            edu_assert(write_callback_state == Write_Callback_Clear);
            write_callback = write_callback_;
            write_callback_state = Write_Callback_Set;
        }
        
        void clear_write_callback() {
            write_callback = [](){};
            write_callback_state = Write_Callback_Clear;
        }

        template<typename T>
            struct ptr_guard_t {
                struct element_guard_t {
                    T element;

                    operator T&() {
                        return element;
                    }

                    T operator=(T x) {
                        element = x;
                        write_callback();
                        return element;
                    }
                };

                static_assert(sizeof(element_guard_t) == sizeof(T), "element guard added padding!");
                static_assert(alignof(element_guard_t) == alignof(T), "element guard added padding!");

                element_guard_t *ptr;
                Buffer buf;
                void *buf_ptr; // detect when changed via cudaMalloc(void **);

                ptr_guard_t() {
                    ptr = nullptr;
                    buf = Buffer::get_uninitialized();
                    buf_ptr = nullptr;
                }

                ptr_guard_t(T *ptr) : ptr_guard_t((element_guard_t*)ptr) {
                }
                ptr_guard_t(element_guard_t *ptr_) : ptr(ptr_) {
                    if(!mem::find_buf(ptr, &buf)) {
                        buf = Buffer::get_universe();
                    }
                    buf_ptr = ptr;
                }
                ptr_guard_t(T *ptr_, Buffer buf_) : ptr_guard_t((element_guard_t *)ptr_, buf_) {
                }
                ptr_guard_t(element_guard_t *ptr_, Buffer buf_) : ptr(ptr_), buf(buf_), buf_ptr(ptr_) {
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

                element_guard_t &operator[](int i) {
                    check_offset(i);
                    return ptr[i];
                }

                element_guard_t &operator*() {
                    return (*this)[0];
                }

                operator T*() {
                    return (T*)ptr;
                }

                ptr_guard_t operator+(int i) {
                    return ptr_guard_t(ptr + i, buf);
                }

                ptr_guard_t operator-(int i) {
                    return *this + (-i);
                }

                ptr_guard_t &operator+=(int i) {
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
            struct array1_guard_t {
                struct element_guard_t {
                    T element;

                    operator T&() {
                        return element;
                    }

                    T operator=(T x) {
                        element = x;
                        write_callback();
                        return element;
                    }
                };
                static_assert(sizeof(element_guard_t) == sizeof(T), "element guard added padding!");
                static_assert(alignof(element_guard_t) == alignof(T), "element guard added padding!");

                element_guard_t data[xlen];

                element_guard_t &operator[](int i) {
                    edu_assert(i >= 0 && i < xlen);
                    return data[i]; 
                }

                operator element_guard_t*() {
                    return data;
                }

                element_guard_t *operator+(int i) {
                    edu_assert(i >= 0 && (i*sizeof(T)) < sizeof(data));
                    return data + i;
                }

                element_guard_t *operator-(int i) {
                    return *this + (-i);
                }
            };

        template<typename T, int xlen, int ylen>
            struct array2_guard_t {
                array1_guard_t<T, ylen> data[xlen];

                array1_guard_t<T, ylen> &operator[](int i) {
                    edu_assert(i >= 0 && i < xlen);
                    return data[i]; 
                }

                operator T*() {
                    return (T*)data;
                }
            };

        template<typename T, int xlen, int ylen, int zlen>
            struct array3_guard_t {
                array2_guard_t<T, ylen, zlen> data[xlen];

                array2_guard_t<T, ylen, zlen> &operator[](int i) {
                    edu_assert(i >= 0 && i < xlen);
                    return data[i]; 
                }

                operator T*() {
                    return (T*)data;
                }
            };

    }
}
