all: build_subdirs

mobile: build_mobile_subdirs

clean: clean_subdirs

SUBDIRS = sparse test

build_subdirs:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done;

build_mobile_subdirs:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir mobile; \
	done;

clean_subdirs:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done;


