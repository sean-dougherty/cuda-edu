#pragma once

#include <cstdlib>
#include <iostream>

namespace edu {
#define edu_err(x...) {std::cerr << __FILE__ << ':' << __LINE__ << ": " << x << std::endl; abort();}
#define edu_errif(expr) if(expr) { edu_err("Failed executing '" << #expr); }
}
