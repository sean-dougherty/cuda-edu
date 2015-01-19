conf=../../../Makefile.conf
include ${conf}

ast_include=-I${PWD}
target=../../bin/educc-ast

${target}: main
	mkdir -p $(shell dirname ${target})
	cp main ${target}

main: main.cpp cxutil.h ${conf} Makefile
	${CXX} $< -o $@ "-DAST_INCLUDE=\"${ast_include}\"" ${AST_CXXFLAGS}

.PHONY: clean
clean:
	rm -f main
	rm -f ${target}