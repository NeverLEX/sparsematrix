PREFIX=/usr/local
ifeq (Makefile.inc, $(wildcard Makefile.inc))
	include Makefile.inc
endif

all:
	make -C src all

clean:
	make -C src clean

