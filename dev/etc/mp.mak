conf=../../Makefile.conf
include ${conf}

srcgen_cu2cpp=.mp.cu-cu2cpp.cpp
srcgen_ast=.mp.cu-ast.cpp
ast_include=-I../../dev/educc/ast

mp: mp.cu $(wildcard ../../dev/include/*.h) $(shell readlink Makefile)
	../../dev/bin/educc-cu2cpp mp.cu > ${srcgen_cu2cpp}
	../../dev/bin/educc-ast ${ast_include} ${srcgen_cu2cpp} > ${srcgen_ast}
	${CXX} -I../../dev/include -o $@ ${srcgen_ast} -g -fopenmp -std=c++11 -lpthread

.PHONY: clean
clean:
	rm -f mp ${srcgen_cu2cpp} ${srcgen_ast}