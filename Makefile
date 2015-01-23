.PHONY: dev clean tests

dev: Makefile.conf
	+ make -C dev/educc/ast
	+ make -C dev/educc/cu2cpp
	@echo "==="
	@echo "=== Build completed successfully"
	@echo "==="

clean: Makefile.conf
	+ make -C dev/educc/ast clean
	+ make -C dev/educc/cu2cpp clean

tests: dev
	+ cd dev/tests && ./run

Makefile.conf:
	./configure