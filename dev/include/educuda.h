#pragma once

#if !defined(EDU_CUDA_SEQUENTIAL) && !defined(EDU_CUDA_THREADS) && !defined(EDU_CUDA_FIBERS)
    #if defined(__linux__)
        #define EDU_CUDA_FIBERS
    #else
        #define EDU_CUDA_THREADS
    #endif
#endif

#if defined(EDU_CUDA_SEQUENTIAL)
    #include <educuda-sequential.h>
#elif defined(EDU_CUDA_THREADS)
    #include <educuda-threads.h>
#elif defined(EDU_CUDA_FIBERS)
    #include <educuda-fibers.h>
#else
    #error "Can't determine cuda driver!"
#endif
