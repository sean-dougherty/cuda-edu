conf=../../../Makefile.conf
include ${conf}

ast_include=-I${PWD}
target=../../bin/educc-ast

${target}: main
	mkdir -p $(shell dirname ${target})
	cp main ${target}

main: main.cpp cxutil.h ${conf} Makefile
	${CXX} ${CXXFLAGS} -o $@ ${CLANG_FLAGS} "-DAST_INCLUDE=\"${ast_include}\"" $< ${CLANG_LIBS}

.PHONY: clean
clean:
	rm -f main
	rm -f ${target}