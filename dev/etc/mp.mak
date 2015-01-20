ifndef EDUHOME
    EDUHOME=../..
endif

conf=${EDUHOME}/Makefile.conf
include ${conf}

srcgen_cu2cpp=.mp.cu-cu2cpp.cpp
srcgen_ast=.mp.cu-ast.cpp
ast_include=-I${EDUHOME}/dev/educc/ast


mp: mp.cu $(wildcard ${EDUHOME}/dev/runtime/*.h) $(shell readlink Makefile)
	${EDUHOME}/dev/bin/educc-cu2cpp mp.cu > ${srcgen_cu2cpp}
	@# Hack: Do a throw-away compile so error messages will be friendly. Once this
	@# phase passes, processes it with ast and then compile a second time.
	@# Won't be necessary if we can get friendlier messages from libclang.
	${CXX} -I${EDUHOME}/dev/runtime -o $@ ${srcgen_cu2cpp} ${RT_CXXFLAGS}
	${EDUHOME}/dev/bin/educc-ast ${ast_include} ${srcgen_cu2cpp} > ${srcgen_ast}
	${CXX} -I${EDUHOME}/dev/runtime -o $@ ${srcgen_ast} ${RT_CXXFLAGS}


.PHONY: clean
clean:
	rm -f mp ${srcgen_cu2cpp} ${srcgen_ast}