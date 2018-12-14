include ../../Makefile.inc

AR=ar
ARFLAGS=-crv
CXX=g++
CXXFLAGS += -I../sparse -I$(BLAS_ROOT)/include
LDFLAGS += -L$(BLAS_ROOT)/lib
LDLIBS +=
EXTRA_LDLIBS =
ADDLIBS = $(BLAS_ROOT)/lib/libopenblas.a ../sparse/libsparsematrix.a

LIBS =
OBJECT =
BINS = blas_test

all: $(BINS) $(LIBS)

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


