#pragma once

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

namespace edu {

#ifdef EDU_CUDA_ERR_THROW
    #define edu_err(x...) {std::stringstream ss; ss << __FILE__ << ":" << __LINE__ << ": " << x << endl; throw runtime_error(ss.str());}
#else

    #define edu_err(x...) {                                                 \
            std::cerr << "ERROR! " << __FILE__ << ':' << __LINE__ << ": " << x << std::endl; \
            abort();                                                        \
        }
#endif

#define edu_panic(x...) {                                               \
        std::cerr << "PANIC! " << __FILE__ << ':' << __LINE__ << ": " << x << std::endl; \
        abort();                                                        \
    }

#define edu_warn(x...) {                                                \
        std::cerr << "Warning! " << __FILE__ << ':' << __LINE__ << ": " << x << std::endl; \
    }

#define edu_errif(expr)                         \
    if(expr) {                                  \
        edu_err("Failed executing '" << #expr); \
}

#define edu_assert(expr)                            \
    if(!(expr)) {                                   \
        edu_err("Assertion failure: " << #expr);    \
    }

    namespace util {
        using namespace std;

        bool file_exists(const string &path) {
            ifstream f(path.c_str()); 
            return f.is_open();
        }

        unsigned parse_uint(const string &str,
                            const string &description,
                            unsigned minval = 0,
                            unsigned maxval = ~0u) {
            typedef long long llong;

            try {
#ifdef __CYGWIN__
	        llong val = atoll(str.c_str());
#else		
                llong val = stoll(str);
#endif		
                if(val >= llong(minval) && val <= llong(maxval)) {
                    return unsigned(val);
                }
            } catch(...) {
            }

            edu_err("Invalid value for " << description << ". Expected integer in [" << minval << ", " << maxval << "].");
        }

        bool ends_with(const string &str, const string &ending) {
            if(str.length() >= ending.length()) {
                return (0 == str.compare(str.length() - ending.length(),
                                         ending.length(),
                                         ending));
            } else {
                return false;
            }
        }

        bool equals(float expected, float actual, float tolerance = 1e-3) {
            if(expected == 0.0) {
                return fabs(expected - actual) < (tolerance);
            } else {
                return fabs(expected - actual) < fabs(tolerance * expected);
            }
        }

        bool equals_abs(float expected, float actual, float tolerance) {
                return fabs(expected - actual) < (tolerance);
        }
    }
}
