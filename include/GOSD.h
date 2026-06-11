#ifndef __GOSD_h
#define __GOSD_h

#include "Head.h"
#include <fftw3-mpi.h>

extern void critical_point();
extern void hiosd_lobpcg_sieh_V0(fftw_complex *x, Option *option,fftw_complex **V0,int k0);
extern void critical_point_v();
#endif
