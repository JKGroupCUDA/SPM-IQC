#ifndef __LOBPCG_h
#define __LOBPCG_h

#include "Head.h"
#include <fftw3-mpi.h>

extern void hv_cam(fftw_complex *phir, fftw_complex *v, fftw_complex *hv);
extern void precond_camnew(fftw_complex *w);
extern void hiosd_lobpcg_initialization(Option *option, int ki);
extern void hiosd_lobpcg_inip(fftw_complex *x ,Option *option );

extern void copy_option_V(Option *option1,Option *option2,int k);
extern void copy_option_V(fftw_complex *v,Option *option2,int k);
extern void copy_option_V(fftw_complex *v,Option *option2,int k);
extern void copy_option_V(fftw_complex **V,Option *option2,int k);
#endif
