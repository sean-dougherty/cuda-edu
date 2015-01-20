# Introduction

cuda-edu is a tool for students of the Coursera *Heterogeneous Parallel Programming* course
that allows for homework assignments to be developed on a local machine without a CUDA GPU.
It should be possible to use exactly the same source code with both cuda-edu and WebGPU. It
is not officially sanctioned by the staff of *Heterogenous Parallel Programming*, it is just
a tool created by a CTA (Community Teaching Assistant).

## What is it?

cuda-edu, essentially, emulates nvcc, libwb, and the CUDA runtimes. It translates your CUDA
code into standard C++ code that can be executed on your CPU.

## Why use it?

You can do local development and use your debugger to step through your code as it executes
on your CPU. Also, cuda-edu injects code that will detect buffer overflows. You program
will trap immediately if you try to dereference your host, device-global, or device-shared
buffers.

## System Requirements

The primary requirements are a C++11 compiler and libclang. Currently, only Linux is
supported. Adding support for Mac should be fairly trivial. A Windows port would require
a non-trivial effort, but shouldn't be too bad.

For Linux installation instructions, please see [Getting Started on Linux](#getting-started-on-linux).


# Dispatchers

cuda-edu has three different thread dispatchers, each with its own strengths. You may find
it beneficial to change dispatchers when trying to debug your code. An overview of how
they work and how to select them follows.

## Sequential Dispatcher

This is the simplest dispatcher, which uses a single OS thread to execute all of your Cuda
threads in sequence. Its main advantage is that it can make debugging really simple. For
example, you can put *printf()* calls in your kernel and not worry about your output getting
garbled. Also, it makes stepping through your code with a debugger really simple. The main
disadvantage is that it cannot support the Cuda *\_\_syncthreads()* call, which means this
dispatcher can only be used with the first couple homework assignments. To activate
the Sequential Dispatcher, you must place a *#define* directive before you include *wb.h*:

```
#define EDU_CUDA_SEQUENTIAL
#include <wb.h>
```

## Threads Dispatcher

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

## Fibers Dispatcher

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

# Getting Started on Linux

*cuda-edu on Linux requires libclang, GNU Make, and a C++ compiler with full c++11 support (e.g. g++ 4.8 or higher).*

## Install git, compiler, and libclang ##
*These instructions are for an Ubuntu system. You may need to do something slightly different
for your distro.*
```
sudo apt-get install git g++ libclang-dev
```

## Download the source
```
git clone https://github.com/sean-dougherty/cuda-edu.git
```

## Build cuda-edu
```
cd cuda-edu
./configure
make
```

The output from this should look similar to the following:
```
[laptop /tmp]$ cd cuda-edu
[laptop /tmp/cuda-edu]$ ./configure
Looking for c++ compiler... OK: /usr/bin/clang++
Searching for */clang-c/Index.h... OK: /usr/lib/llvm-3.4/include
Searching for */libclang.so... OK: /usr/lib/llvm-3.4/lib
generating Makefile.conf...
[laptop /tmp/cuda-edu]$ make
make -C dev/educc/ast
make[1]: Entering directory `/tmp/cuda-edu/dev/educc/ast'
/usr/bin/clang++ main.cpp -o main "-DAST_INCLUDE=\"-I/tmp/cuda-edu\"" -O2 -I/usr/lib/llvm-3.4/include -L/usr/lib/llvm-3.4/lib -lclang -std=c++11 -g -Wall -lrt
mkdir -p ../../bin
cp main ../../bin/educc-ast
make[1]: Leaving directory `/tmp/cuda-edu/dev/educc/ast'
make -C dev/educc/cu2cpp
make[1]: Entering directory `/tmp/cuda-edu/dev/educc/cu2cpp'
/usr/bin/clang++ main.cpp -o main -O2 -std=c++11 -g -Wall -lrt
mkdir -p ../../bin
cp main ../../bin/educc-cu2cpp
make[1]: Leaving directory `/tmp/cuda-edu/dev/educc/cu2cpp'
```

### configure workaround
If the configure step fails, then you can manually configure. Create a *Makefile.conf*:
```
cp Makefile.conf.in Makefile.conf
```
Now edit the portion of Makefile.conf within the "INPUT PARAMETERS" block for your
system by replacing the *\_\_configure...\_\_* text with the appropriate values for your
system. Once you've done that, try executing *make* again.

## Run Unit Tests

Now verify your installation is correct:
```
cd dev/tests
./run
```

After this script runs, you should see a message like the following at the end:
```
Unit tests passed.
```

## Run MP0 (Device Query)
```
cd mp/0
./run
```

This should produce output like the following:
```
../../dev/bin/educc-cu2cpp mp.cu > .mp.cu-cu2cpp.cpp
/usr/bin/clang++ -I../../dev/include -o mp .mp.cu-cu2cpp.cpp -O0 -fopenmp -lpthread -std=c++11 -g -Wall -lrt
../../dev/bin/educc-ast -I../../dev/educc/ast .mp.cu-cu2cpp.cpp > .mp.cu-ast.cpp
/usr/bin/clang++ -I../../dev/include -o mp .mp.cu-ast.cpp -O0 -fopenmp -lpthread -std=c++11 -g -Wall -lrt
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

## Run MP1 Stub (Vector Addition)
The file cuda-edu/mp/1/mp.cu is only the skeleton code found on WebGPU, so it won't succeed
until you've added the necessary code. Even with the code, though, it's possible to compile
and run it. First, let's compile:
```
cd ../1
make
```

Next, let's try running against all the datasets:
```
./run
```
*Note: you don't need to explicitly call 'make'. 'run' will do it for you.*
This should produce output like the following:

```
../../dev/bin/educc-cu2cpp mp.cu > .mp.cu-cu2cpp.cpp
/usr/bin/clang++ -I../../dev/include -o mp .mp.cu-cu2cpp.cpp -O0 -fopenmp -lpthread -std=c++11 -g -Wall -lrt
mp.cu:16:13: warning: unused variable 'deviceOutput' [-Wunused-variable]
    float * deviceOutput;
            ^
mp.cu:15:13: warning: unused variable 'deviceInput2' [-Wunused-variable]
    float * deviceInput2;
            ^
mp.cu:14:13: warning: unused variable 'deviceInput1' [-Wunused-variable]
    float * deviceInput1;
            ^
3 warnings generated.
../../dev/bin/educc-ast -I../../dev/educc/ast .mp.cu-cu2cpp.cpp > .mp.cu-ast.cpp
/usr/bin/clang++ -I../../dev/include -o mp .mp.cu-ast.cpp -O0 -fopenmp -lpthread -std=c++11 -g -Wall -lrt
---
--- TESTING DATASET data/0
---
The input length is 64
ERROR! ../../dev/include/eduwb.h:304: Results mismatch at index 0. Expected 3.6, found 0.
./run: line 21: 12474 Aborted                 (core dumped) $CMD $x
```

You don't have to run against all datasets every time. You can, for example, run only
the third dataset like:

```
./run 3
```

which produces:
```
[laptop /tmp/cuda-edu/mp/1]$ ./run 3
make: `mp' is up to date.
---
--- TESTING DATASET data/3
---
The input length is 100
ERROR! ../../dev/include/eduwb.h:304: Results mismatch at index 0. Expected 103.558, found 0.
./run: line 21: 12486 Aborted                 (core dumped) $CMD $x
```

## Run with Debugger
The *run* script can be told to launch your program with gdb:
```
./run -g 0
```

## Executing without the run script
If you want to run your program without the run script, the usage is:
```
usage: ./mp data/i

example: ./mp data/2
```

So, if you wanted to execute against dataset 0, you would execute:
```
./mp data/0
```

# Getting Started (Windows)

Coming soon.
