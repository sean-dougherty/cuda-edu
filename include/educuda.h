#pragma once

#ifdef EDU_CUDA_SEQUENTIAL
        #include <educuda-sequential.h>
#else
        #include <educuda-threads.h>
#endif
