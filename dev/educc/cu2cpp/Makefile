conf=../../../Makefile.conf
include ${conf}

target=../../bin/educc-cu2cpp

${target}: main
	mkdir -p $(shell dirname ${target})
	cp main ${target}

educc-cu2cpp: main.cpp ${conf} Makefile
	${CXX} ${CXXFLAGS} -o $@ $<

.PHONY: clean
clean:
	rm -f educc-cu2cpp