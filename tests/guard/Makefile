main: main.cpp $(wildcard ../../include/*.h) Makefile
	g++ -g -std=c++11 -I../../include $< -o $@

.PHONY: run
run: main
	./main