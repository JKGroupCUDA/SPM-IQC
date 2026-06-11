#ifndef __BasicOperators_h
#define __BasicOperators_h

#include "Head.h"

extern void FuncsLinear1Cplx(fftw_complex *rslt, int n,
					  const double a1, const fftw_complex *F1);

extern void FuncsLinear2Cplx(fftw_complex *rslt, int n,
					  const double a1, const fftw_complex *F1,
					  const double a2, const fftw_complex *F2);

extern void FuncCplxAddAConst(fftw_complex *rslt, int n,
					   const double a);

extern void FuncRealAddAConst(fftw_complex *rslt, int n,
					   const double a);

extern void Print(fftw_complex *x,int n);
extern void Print(double *x, int n);
extern void Print(fftw_complex *x);
extern void Print(double *x);
extern void Cpy(fftw_complex *a,fftw_complex *b,int n);
extern void norm_fft(fftw_complex *x,int n);
extern double Max(double *x,int n);

extern double mean_Cplx(fftw_complex *src, int n);
extern void FFT_dot_constant(fftw_complex *rslt, int n,
					   double a1);

extern void Dot_constant(double *rslt, int n,double a1);
extern void setCplxZero(fftw_complex rslt);
extern void setCplxZero_V(fftw_complex **W,int k,int cplxDofs);
extern void setCplxZero_v(fftw_complex *x,int cplxDofs);
extern void Orthoi_v2V(fftw_complex *v, fftw_complex **V,int k);
extern void Orthoi_Vi(fftw_complex **V,int k);
extern void get_Merge(double **UU,fftw_complex **V,fftw_complex **W,fftw_complex **Vp, int k);
extern int get_nozero(int *emp,int k);
extern MatrixXd get_mat(double **UU,int *emp,int k,int m);
extern MatrixXd sort_eig(MatrixXd vector, MatrixXd value,int m);
extern void Printf_begin_end(fftw_complex *x);
extern void PV_Cplx(fftw_complex *v, fftw_complex **V,fftw_complex *tmpp,int k);
extern void PV_Cplx(fftw_complex *v, fftw_complex *V,fftw_complex *tmpp,int k);

/*-------Functions nequiring communication -----------------*/
extern double inpfp(fftw_complex *p, fftw_complex *q);
extern double inpfp(double *p, double *q);
extern double nrmp(fftw_complex *p);
extern double nrmp(double *p);
extern double rayleighq(fftw_complex *x, fftw_complex *v);

#endif
