#ifndef __Initialization_h
#define __Initialization_h

#include <fftw3-mpi.h>
#include <math.h>
#include <Head.h>
using namespace std;
extern void memAllocation_all();
extern void memAllocation_local();
extern void LPsystParameter();
extern void get_kindex_all();
extern void get_ikt2_local();
extern void get_ksquare_local();
extern void get_epmckt2_local();

extern double get_GK(double k2);
extern void get_he(fftw_complex *v);

extern void get_initial_value(fftw_complex *v,int choice, double angle);


extern void Distribute_C(int **A, int **a);
extern void Distribute_C(fftw_complex *A, fftw_complex *a);
#endif
