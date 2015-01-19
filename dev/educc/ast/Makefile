conf=../../../Makefile.conf
include ${conf}

target=../../bin/educc-ast

${target}: main
	mkdir -p $(shell dirname ${target})
	cp main ${target}

main: main.cpp cxutil.h ${conf} Makefile
	${CXX} $< -o $@ ${AST_CXXFLAGS}

.PHONY: clean
clean:
	rm -f main
	rm -f ${target}