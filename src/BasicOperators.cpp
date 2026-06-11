#include "Data.h"
#include "BasicOperators.h"
#include "Head.h"
#include "LOBPCG.h"
using namespace std;

void FuncsLinear1Cplx(fftw_complex *rslt, int n,
					  const double a1, const fftw_complex *F1)
{
	for(int i = 0; i < n; i++)
	{
		rslt[i][0] = a1*F1[i][0];
		rslt[i][1] = a1*F1[i][1];
	}
}

void FuncsLinear2Cplx(fftw_complex *rslt, int n,
					  const double a1, const fftw_complex *F1,
					  const double a2, const fftw_complex *F2)
{
	for(int i = 0; i < n; i++)
	{
		rslt[i][0] = a1*F1[i][0] + a2*F2[i][0];
		rslt[i][1] = a1*F1[i][1] + a2*F2[i][1];
	}
}

void FuncCplxAddAConst(fftw_complex *rslt, int n,
					   const double a)
{
	for(int i = 0; i < n; i++)
	{
		rslt[i][0] += a;
		rslt[i][1] += a;
	}
}

void FuncRealAddAConst(fftw_complex *rslt, int n,
					   const double a)
{
	for(int i = 0; i < n; i++)
		rslt[i][0] += a;
}

void Print(double *x,int n)
{
	int s;
	for(int i=0;i<n;i++)
	{
		printf("%.3e\t",x[i]);
		if((i+1)%N==0)
			printf("\n");
	}
}

void Print(fftw_complex *x,int n)
{
	int s;
	for(int i=0;i<n;i++)
	{
		printf("%.3e\t",x[i][0]);
		if((i+1)%N==0)
			printf("\n");
	}
}

void Print(fftw_complex *x)
{
	int s;
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{	
			s = i*N+j;
			printf("(%.3e,%.3e)\t",x[s][0],x[s][1]);
		}
		printf("\n");
	}
}
void Print(double *x)
{
	int s;
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{	
			s = i*N+j;
			printf("%.3e\t",x[s]);
		}
		printf("\n");
	}
}
void Cpy(fftw_complex *a,fftw_complex *b,int n)
{
	//Copy a to b for local data
	for(int i=0;i<n;i++)
	{
		b[i][0] = a[i][0];
		b[i][1] = a[i][1];
	}
}
void norm_fft(fftw_complex *x,int n)
{
	for(int i=0;i<n;i++)
	{
		x[i][0] = x[i][0]/(1.0*CplxDofs);
		x[i][1] = x[i][1]/(1.0*CplxDofs); 
	}
}
double Max(double *x,int n)
{
	double s = 0.0;
	for(int i=0;i<n;i++)
	{
		if(s<x[i])
			s = x[i];
	}
	return s;
}

double mean_Cplx(fftw_complex *src, int n)
{
    double s =0;
    for(int i=0;i<n;i++)
    {
        s = s + src[i][0];
    }
    s = s/n;
    return s;
}

double inpfp(fftw_complex *p, fftw_complex *q)
{
    double s = 0.0,s_all = 0.0;
    for(int i=0;i<CplxDofs_local;i++)
    {
        s = s + p[i][0]*q[i][0];
    }
    s = s/CplxDofs;
	MPI_Allreduce(&s,&s_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    return s_all;
}

double inpfp(double *p, double *q)
{
    double s = 0.0,s_all = 0.0;
    for(int i=0;i<CplxDofs_local;i++)
    {
        s = s + p[i]*q[i];
    }
    s = s/CplxDofs;
	MPI_Allreduce(&s,&s_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    return s_all;
}

double nrmp(fftw_complex *p)
{
    double s = 0.0;
    s = inpfp(p,p);
    s = sqrt(s);
    return s;
}

double nrmp(double *p)
{
    double s = 0.0;
    s = inpfp(p,p);
    s = sqrt(s);
    return s;
}

double rayleighq(fftw_complex *x, fftw_complex *v)
{
    double s1,s2,s;
    fftw_complex *Hv;
    Hv = (fftw_complex *)malloc( sizeof(fftw_complex) *CplxDofs_local);
    hv_cam(x,v,Hv);
    s1 = inpfp(v,Hv);
    s2 = inpfp(v,v);
    s = s1/s2;
    fftw_free(Hv);
    return s;
}

void FFT_dot_constant(fftw_complex *rslt, int n,
					   double a1)
{
	for(int i = 0; i < n; i++)
	{
		rslt[i][0] = a1*rslt[i][0];
		rslt[i][1] = a1*rslt[i][1];
	}
}

void Dot_constant(double *rslt, int n,double a1)
{
	for(int i = 0; i < n; i++)
	{
		rslt[i] = a1*rslt[i];
	}
}

void setCplxZero(fftw_complex rslt)
{
	rslt[0] = 0.0; rslt[1] = 0.0;
}

void setCplxZero_V(fftw_complex **W,int k,int cplxDofs)
{
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<cplxDofs;j++)
        {
            setCplxZero(W[i][j]);
        }
    }
}

void setCplxZero_v(fftw_complex *x,int cplxDofs)
{
    for(int i=0;i<cplxDofs;i++)
    {
        setCplxZero(x[i]);
    }
}

void Orthoi_v2V(fftw_complex *v, fftw_complex **V,int k)
{
    double tmp = 0.0;
    for(int i=0;i<k;i++)
    {
        tmp = inpfp(v,V[i]);
        for(int j=0;j<CplxDofs_local;j++)
        {
            v[j][0] = v[j][0] -tmp*V[i][j][0];
            v[j][1] = 0.0;
        }
    }
}

void Orthoi_Vi(fftw_complex **V,int k)
{
    double tamp = 0.0;
    for(int i=0;i<k;i++)
    {
        tamp = inpfp(V[i],V[k]);
        for(int j=0;j<CplxDofs_local;j++)
        {
            V[k][j][0] = V[k][j][0]-tamp*V[i][j][0];
            V[k][j][1] = 0.0;
        }
    }
}

void get_Merge(double **UU,fftw_complex **V,fftw_complex **W,fftw_complex **Vp, int k)
{
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            UU[i][j] = V[i][j][0];
        }
    }

    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            UU[k+i][j] = W[i][j][0];
        }
    }

    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            UU[2*k+i][j] = Vp[i][j][0];
        }
    }
}

int get_nozero(int *emp,int k)
{
    int iter = 0;
    for(int i=0;i<3*k;i++)
    {
        if(emp[i]==1)
        {
            iter = iter + 1;
        }
    }
    return iter;
}

MatrixXd get_mat(double **UU,int *emp,int k,int m)
{
    MatrixXd UUm = MatrixXd::Zero(m,CplxDofs_local);
    int iter = 0;
    for(int i=0;i<3*k;i++)
    {
        if(emp[i] == 1)
        {
            for(int j=0;j<CplxDofs_local;j++)
                UUm(iter,j) = UU[i][j];
            iter = iter + 1;
        }
    }
    return UUm;
}

MatrixXd sort_eig(MatrixXd vector, MatrixXd value,int m)
{
    MatrixXd value1  = value;
    MatrixXd vector1 = vector;
    double maxi = value.maxCoeff();

    sort(value.data(),value.data()+value.size());
    for(int i=0;i<value.rows();i++)
    {
        for( int j=0;j<value.rows();j++)
        {
            if ( value(i) == value1(j) )
            {
                vector1.col(i) = vector.col(j);
                value1(j) = maxi+1;
                break;
            }
        }
    }
    //cout << value << endl << endl;
    return vector1;
}


void Printf_begin_end(fftw_complex *x)
{
    printf("\n begin:  %.3e \t",x[0][0]);
    printf("end:  %.3e \n",x[CplxDofs_local-1][0]);
}


void PV_Cplx(fftw_complex *v, fftw_complex **V,fftw_complex *tmpp,int k)
{
    setCplxZero_v(tmpp,CplxDofs_local);
    double tmp;
    for(int i=0;i<k;i++)
    {
        tmp = inpfp(v,V[i]);
        for(int j=0;j<CplxDofs_local;j++)
        {
            tmpp[j][0] = tmpp[j][0] + tmp*V[i][j][0];
            tmpp[j][1] = 0.0;
        }
    }
}


void PV_Cplx(fftw_complex *v, fftw_complex *V,fftw_complex *tmpp,int k)
{
	setCplxZero_v(tmpp,CplxDofs_local);
	double tmp;
	tmp = inpfp(v,V);
	for(int j=0;j<CplxDofs_local;j++)
	{
		tmpp[j][0] = tmpp[j][0] + tmp*V[j][0];
		tmpp[j][1] = 0.0;
	}

}


