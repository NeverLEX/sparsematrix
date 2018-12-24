PREFIX=/usr/local
ifeq (Makefile.inc, $(wildcard Makefile.inc))
	include Makefile.inc
endif

all:
	make -C src all

mobile:
	make -C src mobile

clean:
	make -C src clean

