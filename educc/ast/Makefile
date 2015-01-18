llvm_home=/usr/lib/llvm-3.4/
llvm_include=${llvm_home}/include/
llvm_lib=${llvm_home}/lib

educc-ast: main.cpp cxutil.h Makefile
	clang++ -Wall -std=c++11 -g -o $@ -I${llvm_include} -L${llvm_lib} $< -lclang