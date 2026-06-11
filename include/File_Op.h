#ifndef __File_Op_h
#define __File_Op_h

#include <fftw3-mpi.h>
#include <Head.h>

extern void Save_Option_V(Option *option, char *filename);

extern void Save_phi(fftw_complex *phi, char *filename);

extern void load_phi(fftw_complex *x, char *filename);

extern void load_Option_V(Option *option, char *filename);

#endif
