#include <fftw3-mpi.h>
#include "Data.h"
#include "LPsrc.h"
#include <math.h>
#include "BasicOperators.h"
#include "LOBPCG.h"
using namespace std;


void hiosd_lobpcg_initialization(Option *option, int ki)
{
    option->k = ki;
    option->maxiter = 21;
    option->dt = 0.1;
    option->epsf = 1e-7;
    option->betat = 1.0;
    option->betau = 1.0;
    option->outputp = 1;
    option->outputd = 1;

    //printf("\n outputp:  %d",option1->outputp);
    option->V = (fftw_complex **) malloc( sizeof(fftw_complex*)*option->k );
    for(int i=0;i<option->k;i++)
    {
        option->V[i] = (fftw_complex*)malloc(sizeof(fftw_complex) *CplxDofs_local);
    }

    int k;
    k = option->k;
	
    MatrixXd Vm,VM;
	VM = MatrixXd::Random(k,CplxDofs);
    Vm = MatrixXd::Random(k,CplxDofs_local);
	int j1;
	for(int i=0;i<k;i++)
	{
		for(int j=0;j<CplxDofs_local;j++)
		{
			j1 = myid*CplxDofs_local+j;
			Vm(i,j) = VM(i,j1);
		}
	}
    //Vm = MatrixXd::Identity(k,CplxDofs_local);
	//if(myid != 0)
	//{
	//	Vm = 0.0*Vm;
	//}

    double **V;
    V = (double **) malloc(sizeof(double *) *k);
    for(int i=0;i<k;i++)
        V[i] = (double *) malloc(sizeof(double ) *CplxDofs_local);


    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            V[i][j] = Vm(i,j);
        }
    }

    double tmp,tmp1,s1;

	double temp1,temp1_all;
	MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0;i<k;i++)
    {	temp1 = 0.0;
        for(int j=0;j<CplxDofs_local;j++) temp1 = temp1 + V[i][j];
		temp1 = temp1 / (1.0*CplxDofs_local);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&(temp1),&(temp1_all),1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		temp1_all = temp1_all / numprocs;
		// Requires one communication step
        for(int j=0;j<CplxDofs_local;j++)
        {
            V[i][j] = V[i][j] - temp1_all;
        }
    }
    // Orthogonalization //
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<i;j++)
        {
            tmp = inpfp(V[i],V[j]);// Requires communication
            for(int l=0;l<CplxDofs_local;l++)
            {
                V[i][l] = V[i][l] - tmp*V[j][l];
            }
        }
        tmp1 = nrmp(V[i]);// Requires communication
        Dot_constant(V[i], CplxDofs_local,1.0/tmp1);
    }

    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            option->V[i][j][0] = V[i][j];
            option->V[i][j][1] = 0.0;
        }
    }

    for(int i=0;i<k;i++)
    {
        free(V[i]);
    }
    free(V);
}



void hv_cam(fftw_complex *phir, fftw_complex *v, fftw_complex *hv)
{
    fftw_complex *phif,*phir2,*vf,*hvf;
    phif = (fftw_complex *) malloc( sizeof(fftw_complex) *CplxDofs_local);
    phir2= (fftw_complex *) malloc( sizeof(fftw_complex) *CplxDofs_local );
    hvf  = (fftw_complex *) malloc( sizeof(fftw_complex) *CplxDofs_local );
    vf   = (fftw_complex *) malloc( sizeof(fftw_complex) *CplxDofs_local );
	
	Cpy(v,data,CplxDofs_local);
	fftw_execute(plan1);
	norm_fft(data,CplxDofs_local);	
	Cpy(data,vf,CplxDofs_local);
	
    for(int i=0;i<CplxDofs_local;i++)
    {
        hvf[i][0] = epmckt2_local[i]*vf[i][0];
        hvf[i][1] = epmckt2_local[i]*vf[i][1];
    }
    //setCplxZero(hvf[0]);
	Cpy(hvf,data,CplxDofs_local);
	fftw_execute(plan2);
	Cpy(data,hv,CplxDofs_local);

    for(int i=0;i<CplxDofs_local;i++)
    {
		hv[i][0] = hv[i][0] + (- Tau - 2*Gamma*phir[i][0] \
				   + 3*phir[i][0]*phir[i][0]) * v[i][0];
    	hv[i][1] = 0.0;
    }

    double mean = 0.0, mean_all=0.0;
    mean = mean_Cplx(hv, CplxDofs_local);
	MPI_Allreduce(&mean,&mean_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	mean_all = mean_all/numprocs;

    //printf(" mean_all : %.3e",mean_all);
	//printf("\nmyid = %d,\t hv:\n",myid);
	//Print(hv,CplxDofs_local);
	

    for(int i=0;i<CplxDofs_local;i++)
	{
		hv[i][0] = hv[i][0] - mean_all;
	}
	
	//MPI_Barrier(MPI_COMM_WORLD);
	//printf("\nmyid = %d,\t hv:\n",myid);
	//Print(hv,CplxDofs_local);

    fftw_free(phif);
    fftw_free(phir2);
    fftw_free(vf);
    fftw_free(hvf);
}

void precond_camnew(fftw_complex *w)
{
	Cpy(w,data,CplxDofs_local);
	fftw_execute(plan1);
	norm_fft(data,CplxDofs_local);	
	Cpy(data,fftw_Ctmp_local,CplxDofs_local);

    for(int i=0;i<CplxDofs_local;i++)
    {
        fftw_Ctmp_local[i][0] = fftw_Ctmp_local[i][0] * ikt2_local[i];
        fftw_Ctmp_local[i][1] = fftw_Ctmp_local[i][1] * ikt2_local[i];
    }
    //fftw_Ctmp_local[0][0] = 0.0;
	//fftw_Ctmp_local[0][1] = 0.0;

   	Cpy(fftw_Ctmp_local,data,CplxDofs_local);
	fftw_execute(plan2);
	Cpy(data,w,CplxDofs_local);

	double mean = 0.0,mean_all = 0.0;
    mean = mean_Cplx(w, CplxDofs_local);
	//printf("\nmean: %.3e \n",mean);
	MPI_Allreduce(&mean,&mean_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	mean_all = mean_all/numprocs;
	
	for(int i=0;i<CplxDofs_local;i++)
	{
		w[i][0] = w[i][0] - mean_all;
		w[i][1] = 0.0;
	}
    //FuncRealAddAConst(w,CplxDofs_local,-mean_all);
}







void hiosd_lobpcg_inip(fftw_complex *x ,Option *option )
{
    int k;
    k = option->k;
    double alpha[k];
    for(int i=0;i<k;i++) alpha[i] = 0.0;
    fftw_complex **U,**W,**Y,**Vp,**Up,**V;
    V = (fftw_complex **)malloc(sizeof(fftw_complex*) * k);
    U = (fftw_complex **)malloc(sizeof(fftw_complex*) * k);
    W = (fftw_complex **)malloc(sizeof(fftw_complex*) * k);
    Y = (fftw_complex **)malloc(sizeof(fftw_complex*) * k);
    Vp = (fftw_complex **)malloc(sizeof(fftw_complex*) * k);
    Up = (fftw_complex **)malloc(sizeof(fftw_complex*) * k);

    for(int i=0;i<k;i++)
    {
       V[i] = (fftw_complex *)malloc(sizeof(fftw_complex) *CplxDofs_local);
       U[i] = (fftw_complex *)malloc(sizeof(fftw_complex) *CplxDofs_local);
       W[i] = (fftw_complex *)malloc(sizeof(fftw_complex) *CplxDofs_local);
       Y[i] = (fftw_complex *)malloc(sizeof(fftw_complex) *CplxDofs_local);
       Vp[i]= (fftw_complex *)malloc(sizeof(fftw_complex) *CplxDofs_local);
       Up[i]= (fftw_complex *)malloc(sizeof(fftw_complex) *CplxDofs_local);
    }

    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            V[i][j][0] = option->V[i][j][0];
            V[i][j][1] = 0.0;
        }
    }

    setCplxZero_V(W,k,CplxDofs_local);
    setCplxZero_V(Y,k,CplxDofs_local);
    setCplxZero_V(U,k,CplxDofs_local);
    setCplxZero_V(Vp,k,CplxDofs_local);
    setCplxZero_V(Up,k,CplxDofs_local);

    double **UU,**YY;
    UU = (double **)malloc(sizeof(double *) * 3*k);
    YY = (double **)malloc(sizeof(double *) * 3*k);
    for(int i=0;i<3*k;i++)
    {
        UU[i] = (double *)malloc(sizeof(double) *CplxDofs_local);
        YY[i] = (double *)malloc(sizeof(double) *CplxDofs_local);
    }


    int emp[3*k];
    int iter = 0,m;
    int maxiter = option->maxiter;
    double nrmW ,temp1, temp1_all,nrmVpi;


    for(int i=0;i<k;i++)
    {
        Orthoi_Vi(V,i);
        temp1 = nrmp(V[i]);
        FFT_dot_constant(V[i], CplxDofs_local,1.0/temp1);
    }

   // printf("\n x: \n ");
   // Printf_begin_end(x);

   // printf("\n V[0]: \n");
   // Printf_begin_end(V[0]);

    double maxf=0,minf=0;
    while(iter < maxiter)
    {
        for(int i=0;i<3*k;i++) emp[i] = 1;
        for(int i=0;i<k;i++)
        {
            hv_cam(x, V[i], U[i]);
            alpha[i] = inpfp(V[i],U[i]);
			//printf(" \n alpha[%d] = %.3e \n ",i,alpha[i]);
            FuncsLinear2Cplx(W[i], CplxDofs_local,1, U[i],-alpha[i],V[i]);
            //nrmW = nrmp(W[i]);
            //printf("\n  nrmW0[%d] = %.3e \n",i,nrmW);
            precond_camnew(W[i]);
            //nrmW = nrmp(W[i]);
			Orthoi_v2V(W[i],V,k);
            //nrmW = nrmp(W[i]);

			//if(myid == 0)
			//	printf("\n  nrmW0[%d] = %.4e \n",i,nrmW);
            //printf("\n  nrmW0[%d] = %.3e \n",i,nrmW);
            Orthoi_Vi(W,i);
            nrmW = nrmp(W[i]);
            //printf("\n  nrmW[%d] = %.3e \n",i,nrmW);
            if(nrmW > 1e-8)
            {
                FFT_dot_constant(W[i], CplxDofs_local,1.0/nrmW);
                hv_cam(x, W[i], Y[i]);
            }
            else
            {
                setCplxZero_v(W[i],CplxDofs_local);
                setCplxZero_v(Y[i],CplxDofs_local);
                emp[k+i] = 0;
            }
        }

        for(int i=0;i<k;i++)
        {
            Orthoi_v2V(Vp[i],V,k);
            Orthoi_v2V(Vp[i],W,k);
            Orthoi_Vi(Vp,i);
            nrmVpi = nrmp(Vp[i]);
            //printf("\n  nrmVpi[%d] = %.3e \n",i,nrmVpi);

            if(nrmVpi > 1e-8)
            {
                FFT_dot_constant(Vp[i], CplxDofs_local,1.0/nrmVpi);
                hv_cam(x, Vp[i], Up[i]);
            }
            else
            {
                setCplxZero_v(Vp[i],CplxDofs_local);
                setCplxZero_v(Up[i],CplxDofs_local);
                emp[2*k+i] = 0;
            }
        }

        get_Merge(UU,V,W,Vp,k);
        get_Merge(YY,U,Y,Up,k);
        m = get_nozero(emp,k);

        MatrixXd UUm = MatrixXd::Zero(m,CplxDofs_local);
        MatrixXd YYm = MatrixXd::Zero(m,CplxDofs_local);
        MatrixXd Pn = MatrixXd::Zero(m,m);
		MatrixXd Pn_all = MatrixXd::Zero(m,m);

        UUm = get_mat(UU,emp,k,m);
        YYm = get_mat(YY,emp,k,m);
        Pn = UUm*YYm.transpose();

		// Communication is required to sum Pn
		//Pn_all = 1.0*Pn;
		MPI_Barrier(MPI_COMM_WORLD);
		//MPI_Allreduce(&(Pn(0,0)),&(Pn_all(0,0)),m*m,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		for(int i=0;i<m;i++)
		{
			for(int j=0;j<m;j++)
			{
				MPI_Allreduce(&(Pn(i,j)),&(Pn_all(i,j)),1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		Pn_all = (Pn_all.transpose() + Pn_all)*0.5;
        Pn_all = Pn_all/CplxDofs;
		//if(myid ==0)
		//	cout<<"\n Pn_all : \n" << Pn_all << endl<< endl;
		
        EigenSolver<MatrixXd> es(Pn_all);

        MatrixXd value= es.eigenvalues().real();
	    MatrixXd eta= es.eigenvectors().real();

        MatrixXd val_c = es.eigenvalues().imag();
        maxf = abs(  val_c.maxCoeff() );
        minf = abs(  val_c.minCoeff() );
		if(maxf > 1e-22 or minf > 1e-22 )
		{
			//printf("\n Complex characteristic value  \n" );
			break;
		}


        //cout << value<< endl << endl;

        //cout << " m = "<< m<< endl<< endl;
        MatrixXd vector1 = eta;
        vector1 = sort_eig(eta,value,m);
        sort(value.data(),value.data()+value.size());
	    //cout << value<< endl << endl;

        //printf("\n eta : \n");
	    //cout << eta<< endl << endl;

        //printf("\n vector1: \n ");
        //cout << vector1 << endl << endl;
        //eta = vector1;
        for(int i=0;i<k;i++)
        {
            for(int j=0;j<CplxDofs_local;j++)
            {
                Vp[i][j][0] = V[i][j][0];
                Vp[i][j][1] = 0.0;
                Up[i][j][0] = U[i][j][0];
                Up[i][j][1] = 0.0;
            }
        }

        MatrixXd eta1 = vector1.leftCols(k);
        MatrixXd Vm = eta1.transpose()*UUm;


        if(iter % 10 == 0)
        {
			if(myid ==0 )
				cout << "\n"<<iter << " "<< value.transpose().leftCols(k)<<endl<<endl;
            //cout << "C  " << " "<< val_c.transpose()<<endl<<endl;
        }
        for(int i=0;i<k;i++)
        {
            for(int j=0;j<CplxDofs_local;j++)
            {
                V[i][j][0] = Vm(i,j);
                V[i][j][1] = 0.0;
            }
        }

        //printf("\n V[k-1]:  \n");
        //Printf_begin_end(V[k-1]);

//        printf("\n <V[0],V[k-1]> = %.3e",inpfp(V[0],V[k-1]) );
//        printf("\n <V[end],V[end]> = %.3e \n",inpfp(V[k-1],V[k-1]) );

		
		MPI_Barrier(MPI_COMM_WORLD);
        for(int i=0;i<k;i++)
        {
            temp1 = mean_Cplx(V[i], CplxDofs_local);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&(temp1),&(temp1_all),1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			temp1_all = temp1_all / numprocs;
			// Requires one communication step
            for(int j=0;j<CplxDofs_local;j++)
            {
                V[i][j][0] = V[i][j][0] - temp1_all;
                V[i][j][1] = 0.0;
            }
        }

        for(int i=0;i<k;i++)
        {
            //Orthoi_Vi(V,i);
            Orthoi_v2V(V[i],V,i);
            temp1 = nrmp(V[i]);
            FFT_dot_constant(V[i], CplxDofs_local,1.0/temp1);
        }


        //printVect(alpha , k);
        //printf(" \n ");
        iter = iter +1;
		MPI_Barrier(MPI_COMM_WORLD);
    }
   // for(int i=0;i<k;i++)
   // {
   //     printf(" alpha[%d] = %.3e " ,i,rayleighq(x, V[i]) );
   // }
   // cout << endl;

    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            option->V[i][j][0] = V[i][j][0];
            option->V[i][j][1] = 0.0;
        }
    }

    for(int i=0;i<k;i++)
    {
        fftw_free(V[i]);
        fftw_free(U[i]);
        fftw_free(W[i]);
        fftw_free(Y[i]);
        fftw_free(Up[i]);
        fftw_free(Vp[i]);
    }
    fftw_free(V);
    fftw_free(U);
    fftw_free(W);
    fftw_free(Y);
    fftw_free(Up);
    fftw_free(Vp);
}

void copy_option_V(Option *option1,Option *option2,int k)
{
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            option2->V[i][j][0] = option1->V[i][j][0];
            option2->V[i][j][1] = 0.0;
        }
    }
}

void copy_option_V(fftw_complex **V,Option *option2,int k)
{
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<CplxDofs_local;j++)
        {
            option2->V[i][j][0] = V[i][j][0];
            option2->V[i][j][1] = 0.0;
        }
    }
}

void copy_option_V(fftw_complex *v,Option *option2,int k)
{
    for(int j=0;j<CplxDofs_local;j++)
    {
        option2->V[k][j][0] = v[j][0];
        option2->V[k][j][1] = 0.0;
    }
}
