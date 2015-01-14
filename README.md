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

# Tips #

cuda-edu creates an OS thread for every CUDA thread, which can be problematic if you have
a large number of threads per block. When you're developing code, you should consider using
a small number of threads per block, like 4 or 8. Note that only one block has active
threads at a given moment. So, if you have a 32x32 grid with 4 threads per block, you'll
have a max of 4 kernel threads at any moment.
