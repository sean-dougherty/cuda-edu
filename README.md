# Introduction

cuda-edu is a tool for students of the Coursera *Heterogeneous Parallel Programming* course
that allows for homework assignments to be developed on a local machine without a CUDA GPU.
It should be possible to use exactly the same source code with both cuda-edu and WebGPU.


# Getting Started (Linux)

*cuda-edu on Linux requires Python 2.7, GNU Make, and g++ 4.8 or higher.*

## Download the source ##
```
git clone https://github.com/sean-dougherty/cuda-edu.git
```

## Run MP0 (Device Query) ##
```
cd cuda-edu/mp/0
./run
```

This should produce output like the following:
```
../../scripts/cueducc mp.cu
There is 1 device supporting CUDA
Device 0 name: cuda-edu fake device
 Computational Capabilities: 3.0
 Maximum global memory size: 4294770688
 Maximum constant memory size: 65536
 Maximum shared memory size per block: 49152
 Maximum threads per block: 1024
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
```

## Run MP1 Stub (Vector Addition) ##
The file cuda-edu/mp/1/mp.cu is only the skeleton code found on WebGPU, so it won't succeed
until you've added the necessary code. Even with the code, though, it's possible to compile
and run it. First, let's compile:
```
make
```

Next, let's try running against all the datasets:
```
./run
```
*Note: you don't need to explicitly call 'make'. 'run' will do it for you.*
This should produce output like the following:

```
---
--- TESTING DATASET data/0
---
The input length is 64
Results mismatch at index 0. Expected 3.6, found -1.32042e-05.
```

You don't have to run against all datasets every time. You can, for example, run only
the third dataset like:

```
run 3
```

which produces:
```
---
--- TESTING DATASET data/3
---
The input length is 100
Results mismatch at index 0. Expected 103.558, found -1.39195e-05.
```

# Getting Started (Windows)

Coming soon.

# Dispatchers #

cuda-edu has three different thread dispatchers, each with its own strengths. You may find
it beneficial to change dispatchers when trying to debug your code. An overview of how
they work and how to select them follows.

## Sequential Dispatcher ##

This is the simplest dispatcher, which uses a single OS thread to execute all of your Cuda
threads in sequence. Its main advantage is that it can make debugging really simple. For
example, you can put *printf()* calls in your kernel and not worry about your output getting
garbled. Also, it makes stepping through your code with a debugger really simple. The main
disadvantage is that it cannot support the Cuda *__syncthreads()* call, which means this
dispatcher can only be used with the first couple homework assignments. To activate
the Sequential Dispatcher, you must place a *#define* directive before you include *wb.h*:

```
#define EDU_CUDA_SEQUENTIAL
#include <wb.h>
```

## Threads Dispatcher ##

This dispatcher provides proper parallelism, using an OS thread for each Cuda thread. This
means *__syncthreads()* will work. Unfortunately, this can result in *a lot* of OS threads
being created, which will potentially make your program run very slowly and also can make
debugging difficult, since you might see up to 1024 threads listed in your debugger! The
good news is that only a single block is executed at a time, so if you use a small number
of threads per block, then you can keep things manageable. When you're using a proper GPU
you'll typically want to use 512 or more threads per block, but when you're using cuda-edu
with the Threads Dispatcher, you'll probably want to keep your threads per block down around
2 - 16. To activate the Threads Dispatcher, you must place a *#define* directive before you
include *wb.h*:

```
#define EDU_CUDA_THREADS
#include <wb.h>
```

This is the default dispatcher for platforms that don't support the Fibers Dispatcher.
Although it runs slowly and can overload your debugger's list of threads, if you keep
the number of threads in a block down to a small number, it can make debugging more simple
than the Fibers Dispatcher.
You'll be able to see all of the threads from your block and inspect each of their state
while your program is paused by the debugger.

## Fibers Dispatcher ##

This dispatcher, like the Threads Dispatcher, provides real parallelism, allowing
*__synchthreads()* to work. Its main advantages are that it is highly efficient and that
it can handle a large number of threads per block, which means that you can run with the
same number of threads per block as you would with a GPU. The disadvantage is that it
uses wizardry (fibers) to execute multiple Cuda threads on a single OS thread. This means
that debugging can be challenging because you won't be able to see the stack for all the
Cuda threads that are currently active. On the plus side, the Cuda threads in a given
block aren't executed in parallel, so you can step through their execution in your debugger
pretty easily. Just understand that any time *__syncthreads()* is called, there is going
to be a switch to another fiber. To activate the Fibers Dispatcher (if it's available for
your platform), you must place a *#define* directive before you include *wb.h*:

```
#define EDU_CUDA_FIBERS
#include <wb.h>
```

In addition, you may configure the *maximum* number of OS threads used. The actual number
of OS threads may be smaller than what you request. How the actual number is chosen is
decided by your implementation of *OpenMP*. Likely, the number will be limited to the
number of CPU cores. You can use any number greater than 0. Reasonable values would be
in [1,16]. To configure the maximum number of threads, use a *#define* before
you include *wb.h*:

```
#define EDU_CUDA_FIBERS
#define EDU_CUDA_FIBERS_OS_THREADS_COUNT 2
#include <wb.h>
```