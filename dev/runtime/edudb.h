#pragma once

#include <educuda.h>

//------------------------------------------------------------
//---
//--- Utility functions for debugger
//---
//------------------------------------------------------------

bool is_idx(const edu::cuda::uint3 &idx,
            unsigned x, unsigned y, unsigned z) {
    return idx.x == x && idx.y == y && idx.z == z;
}

// Is thread index
bool ist(unsigned x, unsigned y, unsigned z) {
    return is_idx(edu::cuda::current_cuda_thread()->threadIdx,
                  x, y, z);
}

// Is block index
bool isb(unsigned x, unsigned y, unsigned z) {
    return is_idx(edu::cuda::current_cuda_thread()->blockIdx,
                  x, y, z);
}

