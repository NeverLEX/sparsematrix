include ../../Makefile.inc

CXXFLAGS +=
LDFLAGS +=
LDLIBS +=
EXTRA_LDLIBS =
ADDLIBS = 

LIBS = libsparsematrix.a
OBJECT = kernel.o sparse-matrix.o
BINS = kernel_test sparse-matrix_test

all: $(BINS) $(LIBS)

mobile: $(LIBS)

$(BINS):$(OBJECT) $(LIBS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $@.cc $(OBJECT) $(LDLIBS) $(LIBS) $(EXTRA_LDLIBS) $(ADDLIBS)

$(LIBS):$(OBJECT)
	$(AR) $(ARFLAGS) $@ $(OBJECT)

%.o:%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJECT)
	rm -rf $(BINS)
	rm -rf $(LIBS)


