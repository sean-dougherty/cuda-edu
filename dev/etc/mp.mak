mp: mp.cu $(wildcard ../../include/*.h) $(shell readlink Makefile)
	../../scripts/cueducc $<

.PHONY: clean
clean:
	rm -f mp .mp.cu-*.cpp