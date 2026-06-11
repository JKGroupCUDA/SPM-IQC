MPI_ROOT ?= /beegfs/software/mpich/mpich-4.0.3
FFTW_ROOT ?= /beegfs/software/fftw/fftw-3.3.10
EIGEN_ROOT ?= /beegfs/software/eigen/include

CXX = $(MPI_ROOT)/bin/mpicxx
CFLAGS=-c -O2 -W -g
INCLUDE=-I ./include -I$(EIGEN_ROOT)\
	 -I$(MPI_ROOT)/include \
	 -lfftw3_mpi -lfftw3 -I$(FFTW_ROOT)/include
LIBPATH=-L ./lib -L$(MPI_ROOT)/lib\
	 -lfftw3_mpi -lfftw3 -L$(FFTW_ROOT)/lib
OBJECTS=Data Initialization GOSD File_Op My_FFT SPM\
		HISD LOBPCG LPsrc BasicOperators Memfree 
all:spm_yzys
./lib/lib%.so:./src/%.cpp
	mkdir -p lib
	$(CXX) $(CFLAGS) -o $@ $< $(INCLUDE) 
spm_yzys:$(addsuffix .so, $(addprefix ./lib/lib, $(OBJECTS))) main_SPM.cpp
	$(CXX) -g $(INCLUDE) main_SPM.cpp -o $@ $(LIBPATH) $(addprefix -l, $(OBJECTS))
clean: 
	-rm ./lib/*.so
	-rm spm_yzys
