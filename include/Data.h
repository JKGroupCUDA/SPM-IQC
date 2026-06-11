#ifndef __Data_h
#define __Data_h
#include <fftw3-mpi.h>
extern int *NCpt, *K,  dimCpt,dimPhy,max_iter;
extern double L;
extern int **kindex;

extern double  **ProjMatrix;
extern double Gamma,Tau,q,dt;
extern fftw_complex *u_F,*u_R;
extern fftw_complex *data;
extern fftw_complex *u_R_local,*u_F_local,*u2_R_local,*u3_R_local,*u4_R_local;
extern fftw_complex *nln_R_local,*nln_F_local,*Laplas_R_local,*Laplas_F_local;
extern fftw_complex *fftw_Ctmp_local,*fftw_Rtmp_local;

extern fftw_complex **Phi0, **dPhi,  *tv;

extern int **kindex_local;
extern double **PK_local,*k_square_local,*epmckt2_local,*ikt2_local;
extern int myid, numprocs;
extern int N,CplxDofs;
extern double PI;
extern int CplxDofs_local;
extern ptrdiff_t alloc_local, local_n0, local_0_start;

extern int Ini_Phase;

extern fftw_plan plan1,plan2;


struct Option
 {
     int k;
     int maxiter;
     int outputp;
     int outputd;
     double dt;
     double epsf;
     double betat;
     double betau;

     fftw_complex **V;
 };
extern Option *option1,*option2;

#endif
