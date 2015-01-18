.PHONY: dev clean

dev:
	+ make -C dev/educc/ast
	+ make -C dev/educc/cu2cpp

clean:
	+ make -C dev/educc/ast clean
	+ make -C dev/educc/cu2cpp clean
