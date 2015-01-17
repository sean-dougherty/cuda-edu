#pragma once

#include <educuda.h>
#include <eduguard.h>
#include <edumem.h>
#include <eduutil.h>
#include <eduwb.h>

#include <assert.h>

using namespace edu::wb;
using namespace edu::cuda;

#define malloc(len) edu::mem::alloc(edu::mem::MemorySpace_Host, len)
#define free(ptr) edu::mem::dealloc(edu::mem::MemorySpace_Host, ptr)
