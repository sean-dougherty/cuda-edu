conf=../../../Makefile.conf
include ${conf}

target=../../bin/educc-cu2cpp

${target}: main
	mkdir -p $(shell dirname ${target})
	cp main ${target}

main: main.cpp ${conf} Makefile
	${CXX} $< -o $@ ${CU2CPP_CXXFLAGS}

.PHONY: clean
clean:
	rm -f main
	rm -f ${target}