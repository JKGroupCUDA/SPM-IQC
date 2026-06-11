#include "Data.h"
#include <fftw3-mpi.h>
int *NCpt, *K, dimCpt,dimPhy,max_iter;
double L;
int **kindex;
double **ProjMatrix;
double Gamma,Tau,q,dt;
fftw_complex *u_F,*u_R;
fftw_complex *data;
fftw_complex *u_R_local,*u_F_local,*u2_R_local,*u3_R_local,*u4_R_local;
fftw_complex *nln_R_local,*nln_F_local,*Laplas_R_local,*Laplas_F_local;
fftw_complex *fftw_Ctmp_local,*fftw_Rtmp_local;

int  **kindex_local;
double **PK_local,*k_square_local,*epmckt2_local,*ikt2_local;
int myid, numprocs;
int N,CplxDofs;
double PI;
int CplxDofs_local;
ptrdiff_t alloc_local, local_n0, local_0_start;
int Ini_Phase;

fftw_complex **Phi0, **dPhi;
fftw_complex *tv;

fftw_plan plan1,plan2;

Option *option1,*option2;
//ptrdiff_t N0,N1;
