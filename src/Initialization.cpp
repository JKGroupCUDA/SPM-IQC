#include <fftw3-mpi.h>
#include "Data.h"
#include "Initialization.h"
#include <math.h>
#include "My_FFT.h"
using namespace std;
void LPsystParameter()
{
	Ini_Phase = 0;
	N = 256;
	CplxDofs= N*N*N;
	PI = 3.14159265358979311599796346854418516;
		
	//const ptrdiff_t N0=N, N1=N;
	dimCpt=3,dimPhy=3,max_iter =100000,L =26;
	dt=0.3;
	
    	Tau = -0.0061;
	Gamma = 0.30;
	if(myid == 0)
	{
		printf("\nset parameter: Tau = %f, Gamma = %f \n",Tau,Gamma);
		printf("\nset parameter: L = %f, N = %d \n",L,N);
	}
    	q = 0;
	
	ProjMatrix = (double **)malloc(sizeof(double*)*dimPhy);
	for(int i = 0; i < dimPhy; i++)
		ProjMatrix[i] = (double *)malloc(sizeof(double)*dimCpt);
	for(int i = 0; i < dimPhy; i ++)
		for(int j = 0; j < dimCpt; j ++)
			ProjMatrix[i][j] = 0.0;
	
//    for(int i = 0; i < DimPhy; i ++) ProjMatrix[i][i] = 1.0;
	for(int j = 0; j < dimCpt; j++)
	{
		ProjMatrix[j][j] = 1.0/L;
	}    

    	NCpt = (int*) malloc(sizeof(int)*dimCpt);
    	for(int i=0;i<dimCpt;i++) NCpt[i] = N;
}

void memAllocation_local()
{
	kindex_local = (int**)malloc(sizeof(int*)*CplxDofs_local);
	PK_local = (double**) malloc(sizeof(double*) *CplxDofs_local);
    	for(int i=0;i<CplxDofs_local;i++)
	{
        	kindex_local[i] = (int*)malloc(sizeof(int) * dimPhy);
		PK_local[i] = (double*) malloc(sizeof(double) * dimPhy);
	}

	k_square_local = (double*)malloc(sizeof(double)*CplxDofs_local);
	epmckt2_local = (double*)malloc(sizeof(double)*CplxDofs_local);
	ikt2_local = (double*)malloc(sizeof(double)*CplxDofs_local);

	u_F_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	u_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	u2_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	u3_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	u4_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	nln_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	nln_F_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	Laplas_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	Laplas_F_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);

	fftw_Ctmp_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	fftw_Rtmp_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);

	Phi0 = (fftw_complex**)malloc(sizeof(fftw_complex*)*2);
	for(int i=0; i<2; i++)
	{
		Phi0[i] = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	}
	tv = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);

}

void memAllocation_all()
{
	K = (int*) malloc(sizeof(int) *N);
	kindex = (int**)malloc(sizeof(int*)*CplxDofs);
	for(int i=0;i<CplxDofs;i++)
	{
		kindex[i] = (int*)malloc(sizeof(int) * dimPhy);
	}
    
	u_F = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *CplxDofs);
	u_R = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *CplxDofs);
}



void get_kindex_all()
{
    for(int i=0;i<N;i++)
    {
        if(i<N/2)
        {
            K[i] = i;
        }
        else
        {
            K[i] = i-N;
        }
    }
    //K[N/2] = 0.0;
    for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			for(int k=0;k<N;k++)
			{
				kindex[i*N*N+j*N+k][0] = K[i];
				kindex[i*N*N+j*N+k][1] = K[j];
				kindex[i*N*N+j*N+k][2] = K[k];
			}
		}
	}
}

void get_ikt2_local()
{
    for(int i=0;i<CplxDofs_local;i++)
    {
        ikt2_local[i] = 1.0/( epmckt2_local[i] + abs(Tau) + 0.02);
    }
}


void get_ksquare_local()
{
	double tmp = 0.0;
	for(int l=0;l<CplxDofs_local;l++)
	{
		for(int i=0;i<dimPhy;i++)
		{
			tmp = 0.0;
			for(int j=0;j<dimCpt;j++)
			{
				tmp = tmp + ProjMatrix[i][j]*kindex_local[l][j];
			}
			PK_local[l][i] = tmp;
			k_square_local[l] = k_square_local[l] + pow(PK_local[l][i],2);
		}
	}
}

double get_GK(double k2)
{
	double ph = 0.568809;
	double d[5] = {1.969182919080340, -14.949858723012090, 37.445683882581790, -38.839451762550148, 13.999261011773505};
	double d0,d2,d4,d6,d8;
	d0 = d[0],d2 = d[1],d4 = d[2];
	d6 = d[3], d8 = d[4];
							
	double s2 = ph*ph;
	double k4,k6,k8;
	k4 = k2*k2;
	k6 = k4*k2;
	k8 = k6*k2;
	double gk;
	gk = exp(-0.5*k2/s2)*(d0+d2*k2+d4*k4+d6*k6+d8*k8) + 0.08;
	return gk;
}

void get_epmckt2_local()
{
	for (int i=0;i<CplxDofs_local;i++)
	{
		epmckt2_local[i] = get_GK(k_square_local[i]);
	}
}

void Distribute_C(fftw_complex *A, fftw_complex *a)
{
	for (int i = 0; i < local_n0; ++i) 
	{
		for (int j = 0; j < N; ++j)
		{
			for(int k=0;k<N;++k)
			{
				a[i*N*N + j*N + k][0] = A[(local_0_start+i)*N*N + j*N + k][0];
				a[i*N*N + j*N + k][1] = A[(local_0_start+i)*N*N + j*N + k][1];
			}
		}
	}
}

void Distribute_C(int **A, int **a)
{
	for (int i = 0; i < local_n0; ++i) 
	{
		for (int j = 0; j < N; ++j)
		{
			for(int k=0;k<N;++k)
			{
				a[i*N*N + j*N + k][0] = A[(local_0_start+i)*N*N + j*N + k][0];
				a[i*N*N + j*N + k][1] = A[(local_0_start+i)*N*N + j*N + k][1];
				a[i*N*N + j*N + k][2] = A[(local_0_start+i)*N*N + j*N + k][2];
			}
		}
	}
}

void get_he(fftw_complex *v)
{
	int index;
	double h = L / (N*1.0);
	vector<double> xgrids(N);
	for(int i=0;i<N;i++)
	{
		xgrids[i] = i * h;
	}

	vector<double> X3(N * N * N), Y3(N * N * N), Z3(N * N * N);

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			for(int k = 0; k < N; k++)
			{
				index = i * N * N + j * N + k;
				X3[index] = xgrids[i];
				Y3[index] = xgrids[j];
				Z3[index] = xgrids[k];
			}
		}
	}
	double threshold = pow(0.1 * L, 3);
	for(int i=0; i<CplxDofs;i++)
	{
		if (pow(X3[i] - 0.5 * L, 2) + pow(Y3[i] - 0.5 * L, 2) + pow(Z3[i] - 0.5 * L, 2) > threshold)
		{
			v[i][0] = 0.0;
		}
	}
}

void get_initial_value(fftw_complex *u_F,int choice, double angle)
{
	if(myid == 0)
	{
    	printf("\nchoice = %d \t angle = %.3e ",choice,angle);
	}
	int n = 0;
	double l = 0.0;
	char initname[100];
   //Kindex = (int *)malloc(sizeof(int) *24);
	
	if(choice==0)
    {
		if(myid == 0)
        	cout << "\n--------------------DIS----------------------------\n" << endl;
        
		sprintf(initname, "./initData/%s","DIS");
		l = 1;
    }
	else if(choice==1)
    {
		if(myid == 0)
        	cout << "\n--------------------BCC----------------------------\n" << endl;
        
		sprintf(initname, "./initData/%s","BCC");
		l = sqrt(2);
    }
    else if(choice == 2)
    {
		if(myid ==0)
        	cout << "\n--------------------FCC----------------------------\n" << endl;
		
		sprintf(initname, "./initData/%s","FCC");
		l = sqrt(3);
    }
    else if(choice == 3)
    {
		if(myid == 0)
    		cout << "\n--------------------HPC----------------------------\n" << endl;
        
		sprintf(initname, "./initData/%s","HPC");
		l = sqrt(2);
    }
    else if(choice == 4)
    {
		if(myid == 0)
    		cout << "\n--------------------LAM----------------------------\n" << endl;
        sprintf(initname, "./initData/%s","LAM");
		l = 1;
    }
	else if(choice == 5)
    {
		if(myid == 0)
    		cout << "\n--------------------DG----------------------------\n" << endl;
        
		sprintf(initname, "./initData/%s","DG");
		l = sqrt(6);
    }

    for(int i=0;i<CplxDofs;i++)
    {
        u_F[i][0] = 0.0;
        u_F[i][1] = 0.0;
    }
	
	FILE *fp=fopen(initname,"r");
	fscanf(fp,"%d",&n);

	int **fIndex = (int **)malloc(sizeof(int*)*n);
	for(int i=0;i<n;i++)
		fIndex[i] = (int *)malloc(sizeof(int) * (dimCpt+1) );
	
	for(int i=0;i<n;i++)
		for(int j=0;j<(dimCpt+1);j++)
			fscanf(fp,"%d",&(fIndex[i][j]));
	
	int k1,k2,k3;
    double kx,ky,kz;
	for(int i=0;i<n;i++)
	{
		kx = 1.0*fIndex[i][0]/l;
		ky = 1.0*fIndex[i][1]/l;
		kz = 1.0*fIndex[i][2]/l;
		
		k1 = round(kx*L);
		k2 = round(ky*L);
		k3 = round(kz*L);

		if(k1 < 0)
			k1 = N + k1;
		if(k2 < 0)
			k2 = N + k2;
		if(k3 < 0)
			k3 = N + k3;

		u_F[k1*N*N+k2*N + k3][0] = fIndex[i][3]*0.1;
	}
  }
