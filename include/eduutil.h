#pragma once

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace edu {
#define edu_warn(x...) {std::cerr << __FILE__ << ':' << __LINE__ << ": " << x << std::endl;}
#define edu_err(x...) {std::cerr << __FILE__ << ':' << __LINE__ << ": " << x << std::endl; abort();}
#define edu_errif(expr) if(expr) { edu_err("Failed executing '" << #expr); }
    namespace util {
        using namespace std;

        bool file_exists(const string &path) {
            ifstream f(path.c_str()); 
            return f.is_open();
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

        bool equals(float expected, float actual) {
            if(expected == 0.0) {
                return fabs(expected - actual) < (1e-3);
            } else {
                return fabs(expected - actual) < (1e-3 * expected);
            }
        }
    }
}
