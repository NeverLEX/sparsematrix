include ../../Makefile.inc

CXXFLAGS += -I../sparse -I$(BLAS_ROOT)/include
LDFLAGS += -L$(BLAS_ROOT)/lib
LDLIBS +=
EXTRA_LDLIBS =
ADDLIBS = $(BLAS_ROOT)/lib/libopenblas.a ../sparse/libsparsematrix.a

LIBS =
OBJECT =
BINS = blas_test

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


