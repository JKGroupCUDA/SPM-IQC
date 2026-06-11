#ifndef __HISD_h
#define __HISD_h

#include "Head.h"
#include <fftw3-mpi.h>

extern void critical_point_HISD();
extern void critical_point_HISD_v();

extern void hiosd_lobpcg_sieh(fftw_complex *x, Option *option);
#endif
