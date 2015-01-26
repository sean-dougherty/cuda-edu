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
on your CPU. Also, cuda-edu injects code that will detect buffer overflows. Your program
will trap immediately if you try to dereference a bad offset in your host, device-global,
or device-shared buffers.

## System Requirements

The primary requirements are a C++11 compiler and libclang. Currently, Linux, Mac, and Windows
are supported.

For Linux installation instructions, please see [Getting Started on Linux](#getting-started-on-linux).

For Mac installation instructions, please see [Getting Started on Mac](#getting-started-on-mac).

For Window installation instructions, please see [Getting Started on Windows](#getting-started-on-windows).

# Getting Started on Linux

*cuda-edu on Linux requires libclang, GNU Make, and a C++ compiler with full c++11 support (e.g. g++ 4.8 or higher).*

## Install git, compiler, and libclang ##
*These instructions are for an Ubuntu system. You may need to do something slightly different
for your distro.*
```
sudo apt-get install git g++ libclang-dev
```

The remaining steps are in the section [Getting Started on POSIX](#getting-started-on-posix).

# Getting Started on Mac

cuda-edu on Mac requires that the XCode Command Line Tools be installed. I'm no Mac expert,
but I was able to install them by typing the following in a Terminal:
```
xcode-select --install
```

If that doesn't work for you, a quick google search should tell you what you need to do. The
remainder of the instructions assume that you've got a Terminal open. Please see
[Getting Started on POSIX](#getting-started-on-posix).

# Getting Started on Windows

cuda-edu on Windows requires that Cygwin be installed. If demand exists, a port to Visual
Studio could be made.

First, install [Cygwin](https://cygwin.com/install.html) (direct link to 32-bit installer: [installer](https://cygwin.com/setup-x86.exe)).
Only the 32-bit version has been tested, so it is recommended you stick with that for now.
Once you have gone through the installation steps of choosing a server and selecting an install directory,
you'll be allowed to select which packages to install. At a minimum, you'll need to install:
g++, libclang-devel, make, and git. It is also recommended that you install gdb for debugging. It
is recommended that you select the option to create a shortcut in your start menu.

Once you have installed those packages, you should be able to start the *Cygwin Terminal* from
your start menu. Once that executes, you should see a terminal application. From this point,
you can follow the generic POSIX instructions. Please see [Getting Started on POSIX](#getting-started-on-posix).

# Getting Started on POSIX

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
[laptop ~/cta]$ cd cuda-edu
[laptop ~/cta/cuda-edu]$ ./configure
Looking for c++ compiler...OK: clang++
Searching for libclang headers... OK: /usr/lib/llvm-3.4/include/clang-c/Index.h
Searching for libclang library... OK: /usr/lib/llvm-3.4/lib/libclang.so.1
generating Makefile.conf...OK
===
=== Configure completed successfully
===
[laptop ~/cta/cuda-edu]$ make
make -C dev/educc/ast
make[1]: Entering directory `/home/dougher1/cta/cuda-edu/dev/educc/ast'
clang++ main.cpp -o main -O2 -I/usr/lib/llvm-3.4/include /usr/lib/llvm-3.4/lib/libclang.so.1 -std=c++11 -g -Wall -lrt -lpthread
mkdir -p ../../bin
cp main ../../bin/educc-ast
make[1]: Leaving directory `/home/dougher1/cta/cuda-edu/dev/educc/ast'
make -C dev/educc/cu2cpp
make[1]: Entering directory `/home/dougher1/cta/cuda-edu/dev/educc/cu2cpp'
clang++ main.cpp -o main -O2 -std=c++11 -g -Wall -lrt -lpthread
mkdir -p ../../bin
cp main ../../bin/educc-cu2cpp
make[1]: Leaving directory `/home/dougher1/cta/cuda-edu/dev/educc/cu2cpp'
===
=== Build completed successfully
===
```

### configure workaround
If the configure step fails, then you can manually configure. Create a *Makefile.conf*:
```
cp dev/etc/Makefile.conf.in Makefile.conf
```
Now edit the portion of Makefile.conf within the "INPUT PARAMETERS" block for your
system by replacing the *\_\_configure...\_\_* text with the appropriate values for your
system. Once you've done that, try executing *make* again.

## Run Unit Tests

Now verify your installation is correct:
```
make tests
```

After this script runs, you should see a message like the following at the end:
```
===
=== Unit tests passed
===
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
./run all
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

When debugging, you may wish to have only a single OS thread executing your
kernel. To do this, use the -s flag:
```
./run -gs 0
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
