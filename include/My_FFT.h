#ifndef __My_FFT_h
#define __My_FFT_h

#include <BasicOperators.h>
#include <Data.h>
#include <fftw3-mpi.h>
#include <Head.h>


extern void my_fft(fftw_complex *x_R_local, fftw_complex *x_F_local);

extern void my_ifft(fftw_complex *x_F_local, fftw_complex *x_R_local);

extern void My_mpi_fft_set_up();
#endif
