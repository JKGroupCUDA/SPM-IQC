#include <fftw3-mpi.h>
#include "Data.h"
#include "LPsrc.h"
#include <math.h>
#include "BasicOperators.h"
#include "LOBPCG.h"
#include "SPM.h"
#include <algorithm>
#include "File_Op.h"
#include "Initialization.h"
#include "My_FFT.h"
#include "SPM.h"
using namespace std;

void get_Tv(fftw_complex **Phi, fftw_complex *v)
{
	double d;
	for(int i=0;i<CplxDofs_local;i++)
	{
		//printf("\n go go go \n ");
		v[i][0] = Phi[1][i][0] - Phi[0][i][0];
		v[i][1] = 0.0;
	}

	d = nrmp(v);
	
	for(int i=0;i<CplxDofs_local;i++)
	{
		v[i][0] =  v[i][0] / d;
		v[i][1] = 0.0;
	}
}

void Natural_Forces(fftw_complex *x,fftw_complex *v)
{

	fftw_complex *rhoCplx1,*f,*tmpp,*tmppF,*nlnF,*nlnR;
	rhoCplx1 = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	f = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	tmpp = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	tmppF = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	nlnR = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	nlnF = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	
	setCplxZero_v(rhoCplx1,CplxDofs_local);
	setCplxZero_v(tmpp,CplxDofs_local);
	setCplxZero_v(f,CplxDofs_local);
	setCplxZero_v(tmppF,CplxDofs_local);
	setCplxZero_v(nlnR,CplxDofs_local);
	setCplxZero_v(nlnF,CplxDofs_local);

	double bta =1.0,mean,mean_all;
	ngrad_cam(x, f);
	PV_Cplx(f,v,tmpp,1);
	//FFT_dot_constant(tmpp,CplxDofs_local,2.0);
	
	for(int i=0;i<CplxDofs_local;i++)
	{
		u2_R_local[i][0] = x[i][0]*x[i][0];
		u2_R_local[i][1] = 0.0;

		u3_R_local[i][0] = u2_R_local[i][0]*x[i][0];
		u3_R_local[i][1] = 0.0;
	}

	my_fft(x, rhoCplx1);
	
	for(int i=0;i<CplxDofs_local;i++)
	{
		nlnR[i][0] = Tau*x[i][0] + Gamma*u2_R_local[i][0] - u3_R_local[i][0];
		nlnR[i][1] = 0.0;
	}

	mean = mean_Cplx(nlnR, CplxDofs_local);
	MPI_Allreduce(&mean,&mean_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	mean_all = mean_all/numprocs;
	FuncRealAddAConst(nlnR,CplxDofs_local,-mean_all);

	my_fft(tmpp,tmppF);
	my_fft(nlnR, nlnF);

	for(int i=0;i<CplxDofs_local;i++)
	{
		rhoCplx1[i][0]=( rhoCplx1[i][0]+bta*nlnF[i][0] - bta*tmppF[i][0])/(1+bta*epmckt2_local[i]);
		rhoCplx1[i][1]=( rhoCplx1[i][1]+bta*nlnF[i][1] - bta*tmppF[i][1])/(1+bta*epmckt2_local[i]);
	}

	if(myid == 0)
		setCplxZero(rhoCplx1[0]);

	my_ifft(rhoCplx1, x);

	fftw_free(rhoCplx1);
	fftw_free(f);
	fftw_free(tmpp);
	fftw_free(tmppF);
	fftw_free(nlnF);
	fftw_free(nlnR);
}



void MinMax_Forces(fftw_complex *x,fftw_complex *v)
{

	fftw_complex *rhoCplx1,*f,*tmpp,*tmppF,*nlnF,*nlnR;
	rhoCplx1 = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	f = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	tmpp = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	tmppF = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	nlnR = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	nlnF = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	
	setCplxZero_v(rhoCplx1,CplxDofs_local);
	setCplxZero_v(tmpp,CplxDofs_local);
	setCplxZero_v(f,CplxDofs_local);
	setCplxZero_v(tmppF,CplxDofs_local);
	setCplxZero_v(nlnR,CplxDofs_local);
	setCplxZero_v(nlnF,CplxDofs_local);

	double bta = 1.0,mean,mean_all;
	ngrad_cam(x, f);
	PV_Cplx(f,v,tmpp,1);
	FFT_dot_constant(tmpp,CplxDofs_local,2.0);
	
	for(int i=0;i<CplxDofs_local;i++)
	{
		u2_R_local[i][0] = x[i][0]*x[i][0];
		u2_R_local[i][1] = 0.0;

		u3_R_local[i][0] = u2_R_local[i][0]*x[i][0];
		u3_R_local[i][1] = 0.0;
	}

	my_fft(x, rhoCplx1);
	
	for(int i=0;i<CplxDofs_local;i++)
	{
		nlnR[i][0] = Tau*x[i][0] + Gamma*u2_R_local[i][0] - u3_R_local[i][0];
		nlnR[i][1] = 0.0;
	}

	mean = mean_Cplx(nlnR, CplxDofs_local);
	MPI_Allreduce(&mean,&mean_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	mean_all = mean_all/numprocs;
	FuncRealAddAConst(nlnR,CplxDofs_local,-mean_all);

	my_fft(tmpp,tmppF);
	my_fft(nlnR, nlnF);

	for(int i=0;i<CplxDofs_local;i++)
	{
		rhoCplx1[i][0]=( rhoCplx1[i][0]+bta*nlnF[i][0] - bta*tmppF[i][0])/(1+bta*epmckt2_local[i]);
		rhoCplx1[i][1]=( rhoCplx1[i][1]+bta*nlnF[i][1] - bta*tmppF[i][1])/(1+bta*epmckt2_local[i]);
	}

	if(myid == 0)
		setCplxZero(rhoCplx1[0]);

	my_ifft(rhoCplx1, x);

	fftw_free(rhoCplx1);
	fftw_free(f);
	fftw_free(tmpp);
	fftw_free(tmppF);
	fftw_free(nlnF);
	fftw_free(nlnR);
}

void Spring_Pairing(fftw_complex **Phi, double maxiter, double tol)
{
	double ks, spring_d, dt;
	double res,resi;
	double dd,kd;
	int iter;

	dt = 0.001;
	ks = 1;
	spring_d = 0.001;
	res = 1e-6;
	iter = 0;

	fftw_complex *Fs, *f, *dphi;
	Fs = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	f = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	dphi = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
	setCplxZero_v(Fs,CplxDofs_local);
	setCplxZero_v(f,CplxDofs_local);
	
	//printf("\n go go go \n ");
	while (res > tol && iter < maxiter )
	{
		iter = iter +1;
		get_Tv(Phi,tv);
	
		for(int i=0;i<2;i++)
		{
			Natural_Forces(Phi[i],tv);
		}
		FuncsLinear2Cplx(dphi,CplxDofs_local,1,Phi[1],-1,Phi[0]);
		dd = nrmp(dphi);

		for( int i=0; i<2; i++)
		{
			if(i==0)
			{
				kd = ks*( dd - spring_d );
				FuncsLinear2Cplx(Fs,CplxDofs_local,ks,Phi[1],-ks,Phi[0]);
			}
			else
			{
				kd = ks*( dd - spring_d );
				FuncsLinear2Cplx(Fs,CplxDofs_local,ks,Phi[0],-ks,Phi[1]);
			}
			FuncsLinear2Cplx(Phi[i],CplxDofs_local,1,Phi[i],dt,Fs);
		}

		get_Tv(Phi,tv);
		
		res = 0.0;
		for(int i=0;i<2;i++)
		{
			MinMax_Forces(Phi[i],tv);
			ngrad_cam(Phi[i],f);
			resi = nrmp( f );
			if ( res < resi )
			{
				res = resi;
			}
		}


		for( int i=0; i<2; i++)
		{
			if(i==0)
			{
				kd = ks*( dd - spring_d );
				FuncsLinear2Cplx(Fs,CplxDofs_local,ks,Phi[1],-ks,Phi[0]);
			}
			else
			{
				kd = ks*( dd - spring_d );
				FuncsLinear2Cplx(Fs,CplxDofs_local,ks,Phi[0],-ks,Phi[1]);
			}
			FuncsLinear2Cplx(Phi[i],CplxDofs_local,1,Phi[i],dt,Fs);
		}
		
		if(myid == 0)
		{
			if(iter % 10 == 0)
			{
				printf("\n iter = %d , res =  %.3e \n",iter,res);
			}
		}

	}

	get_Tv(Phi,tv);
	fftw_free(Fs);
	fftw_free(f);
	fftw_free(dphi);

}

void Gs_gri_nuc(fftw_complex *v_loc, fftw_complex *phi_loc)
{
	double h = 2.0/N;
	double alpha = 18;
	double x,y,z;
	double *r_sq,*r_sq_loc;
	r_sq     = (double*)malloc(sizeof(double)*CplxDofs);
	r_sq_loc = (double*)malloc(sizeof(double)*CplxDofs_local);

	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				x = -1 + i*h;
				y = -1 + j*h;
				z = -1 + k*h;
				r_sq[i*N*N+j*N+k] = pow(x,2)+pow(y,2)+pow(z,2);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatter(r_sq,alloc_local,MPI_DOUBLE,r_sq_loc,alloc_local,MPI_DOUBLE,0,MPI_COMM_WORLD);

	for(int i=0;i<CplxDofs_local;i++)
	{
		v_loc[i][0] = exp(-alpha*r_sq_loc[i])*phi_loc[i][0];
		v_loc[i][1] = 0.0;
	}
	

	free(r_sq);
	free(r_sq_loc);
}

















