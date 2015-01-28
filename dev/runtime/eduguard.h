#pragma once

#include <edumem.h>

#include <functional>

namespace edu {
    namespace guard {
        using namespace std;
        using edu::mem::Buffer;

        //------------------------------------------------------------
        //---
        //--- Write Callback Mechanism
        //---
        //--- The write callback gives us a hook for when memory is
        //--- updated, which allows us to coordinate threads in a
        //--- warp.
        //------------------------------------------------------------
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

        //------------------------------------------------------------
        //---
        //--- STRUCT dptr_guard_t
        //---
        //--- double pointer guard (void **), which gives us a
        //--- callback hook for things like cudaMalloc()
        //------------------------------------------------------------
        template<typename T>
            struct dptr_guard_t {
                T **dptr;
                function<void()> callback;

                dptr_guard_t(T **_dptr, function<void()> _callback)
                : dptr(_dptr)
                , callback(_callback) {
                }

                ~dptr_guard_t() {
                    callback();
                }

                operator T**() {
                    return dptr;
                }

                operator void**() {
                    return (void **)dptr;
                }
            };

        //------------------------------------------------------------
        //---
        //--- STRUCT ptr_guard_t
        //---
        //--- Guards a pointer from buffer overflow
        //------------------------------------------------------------
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

                ptr_guard_t() {
                    ptr = nullptr;
                    buf = Buffer::get_uninitialized();
                }

                ptr_guard_t(T *ptr_) : ptr((element_guard_t*)ptr_) {
                    if(!mem::find_buf(ptr, &buf)) {
                        buf = Buffer::get_universe();
                    }
                }

                ptr_guard_t(T *ptr_, Buffer buf_)
                    : ptr_guard_t((element_guard_t *)ptr_, buf_) {
                }
                ptr_guard_t(element_guard_t *ptr_, Buffer buf_)
                    : ptr(ptr_), buf(buf_) {
                }

                ptr_guard_t &operator=(T *ptr_) {
                    this->~ptr_guard_t();
                    return *(new (this) ptr_guard_t(ptr_));
                }

                dptr_guard_t<T> operator&() {
                    return dptr_guard_t<T>((T**)&ptr,
                                           [this]() {
                                               mem::find_buf(ptr, &buf);
                                           });
                }
                

                void check_offset(int i) {
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

                template<typename U>
                operator U*() {
                    return (U*)ptr;
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


        //------------------------------------------------------------
        //---
        //--- STRUCT array1_guard
        //---
        //--- guard for 1-dimensional array.
        //------------------------------------------------------------
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

        //------------------------------------------------------------
        //---
        //--- STRUCT array2_guard
        //---
        //--- guard for 2-dimensional array.
        //------------------------------------------------------------
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

        //------------------------------------------------------------
        //---
        //--- STRUCT array3_guard
        //---
        //--- guard for 3-dimensional array.
        //------------------------------------------------------------
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
