include ../../../Makefile.conf

run: educc-ast
	./run "${AST_CXXINCLUDES}"

.PHONY: educc-ast clean
clean:
	rm -rf tmp

educc-ast:
	cd ../../educc/ast && make