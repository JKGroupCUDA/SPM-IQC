#ifndef __SPM_h
#define __SPM_h

#include "Head.h"
#include <fftw3-mpi.h>

extern void get_Tv(fftw_complex **Phi,fftw_complex *v);
extern void Natural_Forces(fftw_complex *x,fftw_complex *v);
extern void Spring_Pairing(fftw_complex **Phi, double maxiter, double tol);
extern void Gs_gri_nuc(fftw_complex *v_loc, fftw_complex *phi_loc);

#endif
