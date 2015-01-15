#pragma once

#include <educuda.h>
#include <edumem.h>
#include <eduutil.h>
#include <eduwb.h>

using namespace edu::wb;
using namespace edu::cuda;

#define malloc(len) edu::mem::alloc(edu::mem::MemorySpace_Host, len)
#define free(ptr) edu::mem::dealloc(ptr)
