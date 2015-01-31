# Introduction

cuda-edu is a tool for students of the Coursera *Heterogeneous Parallel Programming* course
that allows for homework assignments to be developed on a local machine without a CUDA GPU.
It should be possible to use exactly the same source code with both cuda-edu and WebGPU. It
is not officially sanctioned by the staff of *Heterogenous Parallel Programming*. It is just
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

## Getting Started

Installation instructions are hosted on the [Wiki](https://github.com/sean-dougherty/cuda-edu/wiki).
Please see the page for your OS:

* [Linux installation](https://github.com/sean-dougherty/cuda-edu/wiki/Getting-Started-on-Linux)

* [Mac installation](https://github.com/sean-dougherty/cuda-edu/wiki/Getting-Started-on-Mac)

* [Windows installation](https://github.com/sean-dougherty/cuda-edu/wiki/Getting-Started-on-Windows)
