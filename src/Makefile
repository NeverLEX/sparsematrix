all: build_subdirs

clean: clean_subdirs

SUBDIRS = sparse test

build_subdirs:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done;

clean_subdirs:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done;


