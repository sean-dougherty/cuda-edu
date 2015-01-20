conf=../../../Makefile.conf
include ${conf}

main: main.cpp $(wildcard ../../runtime/*.h) Makefile ${conf}
	${CXX} -I../../runtime $< -o $@ ${RT_CXXFLAGS}

.PHONY: run clean
run: main
	./main

clean:
	rm -f main