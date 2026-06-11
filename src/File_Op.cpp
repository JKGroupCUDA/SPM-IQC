#include <fftw3-mpi.h>
#include "Data.h"
#include "Initialization.h"
#include "BasicOperators.h"
#include "File_Op.h"
#include "Head.h"

void Save_Option_V(Option *option, char *filename)
{
	printf("\n------------Save---V----------------------------\n");
	fftw_complex *v;
	v = (fftw_complex *)malloc(sizeof(fftw_complex) * CplxDofs);	
	
	FILE* fp;
	if(myid == 0)
	{
		fp = fopen(filename,"w");
	}
	for(int i=0;i<option->k;i++)
	{
		MPI_Gather(option->V[i],alloc_local,MPI_C_DOUBLE_COMPLEX,v,alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
		if(myid == 0)
		{
			for(int j=0;j<CplxDofs;j++)
			{
				fprintf(fp,"%.16e \n",v[j][0]);
			}
		}
	}
	if(myid == 0)
	{
		fclose(fp);
	}
	fftw_free(v);
}

void Save_phi(fftw_complex *phi, char *filename)
{
	printf("\n--------------save-----------------\n");
	FILE* fp;
	fp = fopen(filename,"w");
	for(int i=0;i<CplxDofs;i++)
	{
		fprintf(fp,"%.16e \n",phi[i][0]);
	}
	fclose(fp);
}

void load_phi(fftw_complex *x, char *filename)
{
	printf("\n -----------loading------------------");
	FILE *fp;
	fp = fopen(filename,"r");
	for(int i=0;i<CplxDofs;i++)
	{
		fscanf(fp,"%lf",&(x[i][0]));
		x[i][1] = 0.0;
	}
	fclose(fp);
}

void load_Option_V(Option *option, char *filename)
{
	printf("\n-----------------load-V----------------------");
	fftw_complex *v;
	v = (fftw_complex *)malloc(sizeof(fftw_complex) * CplxDofs);	
	setCplxZero_v(v,CplxDofs);
	FILE* fp;
	if(myid == 0)
	{
		fp = fopen(filename,"r");
	}
	for(int i=0;i<option->k;i++)
	{
		if(myid == 0)
		{
			for(int j=0;j<CplxDofs;j++)
			{
				fscanf(fp,"%lf",&(v[j][0]));
			}
		}
		MPI_Scatter(v,alloc_local,MPI_C_DOUBLE_COMPLEX,option->V[i],alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
	}
	if(myid == 0)
	{
		fclose(fp);
	}
	fftw_free(v);
}

