#include <fftw3-mpi.h>
#include "Data.h"
#include "LPsrc.h"
#include <math.h>
#include "BasicOperators.h"
#include "My_FFT.h"

double get_Ene()
{
	double e=0,E1=1,E2=1;
    for(int i =0;i<CplxDofs_local;i++)
    {
        u2_R_local[i][0] = u_R_local[i][0]*u_R_local[i][0];
        u2_R_local[i][1] = 0.0;
        u3_R_local[i][0] = u2_R_local[i][0]*u_R_local[i][0];
        u3_R_local[i][1] = 0.0;
        u4_R_local[i][0] = u3_R_local[i][0]*u_R_local[i][0];
        u4_R_local[i][1] = 0.0;
		nln_R_local[i][0]  = -0.5*Tau*u2_R_local[i][0] - Gamma*u3_R_local[i][0]/3 + 0.25*u4_R_local[i][0];	
        nln_R_local[i][1]  = 0.0;
    }

	my_fft(nln_R_local,nln_F_local);	
    E2 = nln_F_local[0][0];
    //printf("\nEnergy 2: %.8e\n",E2);

    for(int i =0;i<CplxDofs_local;i++)
    {
        Laplas_F_local[i][0] = u_F_local[i][0]*u_F_local[i][0] + u_F_local[i][1]*u_F_local[i][1];
		Laplas_F_local[i][0] = Laplas_F_local[i][0] * epmckt2_local[i];
		Laplas_F_local[i][1] = 0.0;
    }

	my_ifft(Laplas_F_local,Laplas_R_local);
	E1 = 0.5*Laplas_R_local[0][0];
	e = E1+E2;
	return e;
}

void get_min_u_F()
{
	double res=1.0,res0;
    int iter =0;
	double E=0;
	fftw_complex *u_F1_local,*f_R_local,*f_F_local;
	u_F1_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	f_F_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);
	f_R_local = (fftw_complex*)malloc(sizeof(fftw_complex)*CplxDofs_local);

	double *grad_local;	
	grad_local = (double*)malloc(sizeof(double)*CplxDofs_local);
    while(res > 1e-16 && iter<max_iter)
    {
		E = get_Ene();
		MPI_Bcast ( &E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
        for(int i =0;i<CplxDofs_local;i++)
        {
			f_R_local[i][0]  = Tau*u_R_local[i][0] + Gamma*u2_R_local[i][0] 
				    - u3_R_local[i][0];
        	f_R_local[i][1]  = 0.0;
        }

		my_fft(f_R_local,f_F_local);

		for(int i=0;i<CplxDofs_local;i++)
		{
			u_F1_local[i][0] = (dt*f_F_local[i][0]+u_F_local[i][0]) / (1+dt*epmckt2_local[i]);
			u_F1_local[i][1] = (dt*f_F_local[i][1]+u_F_local[i][1]) / (1+dt*epmckt2_local[i]);
		}
		if(myid == 0)
		{
			//printf("\n%d \t Energy: %.15e\n",iter,E);
			u_F1_local[0][0] = 0.0;
			u_F1_local[0][1] = 0.0;
		}
		
		for(int i=0;i<CplxDofs_local;i++)
		{
			grad_local[i] = (u_F1_local[i][0]-u_F_local[i][0])*(u_F1_local[i][0]-u_F_local[i][0]);
			grad_local[i] = 1.0*grad_local[i];
			grad_local[i] = sqrt(grad_local[i]);
		}

		res = Max(grad_local,CplxDofs_local);

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&res, &res0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		
		if(myid ==0)
			res = res0;

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast ( &res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );

		//MPI_Barrier(MPI_COMM_WORLD);
		if(myid ==0)
		{
			if(iter%10 ==0)
				printf("\nmyid : %d ,\t iter : %d \t res: %.3e \t E: %.3e \t",myid,iter,res,E);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
        for(int i=0;i<CplxDofs_local;i++)
        {
            u_F_local[i][0] = u_F1_local[i][0];
            u_F_local[i][1] = u_F1_local[i][1];
        }

		my_ifft(u_F_local,u_R_local);
		        
		iter = iter +1;
    }

	fftw_free(u_F1_local);
	fftw_free(f_F_local);
	fftw_free(f_R_local);
	fftw_free(grad_local);
}

void ngrad_cam(fftw_complex *phir, fftw_complex *gradr)
{
    fftw_complex *phif,*gradf;
    phif = (fftw_complex *) malloc(sizeof(fftw_complex) * CplxDofs_local);
    gradf= (fftw_complex *) malloc(sizeof(fftw_complex) * CplxDofs_local);
	
	Cpy(phir,data,CplxDofs_local);
	fftw_execute(plan1);
	norm_fft(data,CplxDofs_local);
	Cpy(data,phif,CplxDofs_local);
	if(myid == 0)
	{
		setCplxZero(phif[0]);
	}
      
	for(int i = 0; i < CplxDofs_local; i++)
	{
        //setCplxZero(gradf[i]);
		gradf[i][0] = epmckt2_local[i] * phif[i][0];
		gradf[i][1] = epmckt2_local[i] * phif[i][1];
	}

	Cpy(gradf,data,CplxDofs_local);
	fftw_execute(plan2);
	Cpy(data,gradr,CplxDofs_local);
    //q = fftw_plan_dft( DimCpt,NCpt,gradf,gradr,FFTW_BACKWARD,FFTW_ESTIMATE);
    //fftw_execute(q);

    for(int i=0;i<CplxDofs_local;i++)
    {
        u2_R_local[i][0] = phir[i][0]*phir[i][0];
        u2_R_local[i][1] = 0.0;

        u3_R_local[i][0] = u2_R_local[i][0]*phir[i][0];
        u3_R_local[i][1] = 0.0;
    }

    for(int i=0;i<CplxDofs_local;i++)
    {
		gradr[i][0] = gradr[i][0] - Tau*phir[i][0] - \
					  Gamma*u2_R_local[i][0] + u3_R_local[i][0];
        gradr[i][1] = 0.0;
    }

    double mean = 0.0,mean_all;
    mean = mean_Cplx(gradr, CplxDofs_local);
	MPI_Allreduce(&mean,&mean_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	mean_all = mean_all/numprocs;
    FuncRealAddAConst(gradr,CplxDofs_local,-mean_all);

    FFT_dot_constant(gradr,CplxDofs_local,-1.0);

    fftw_free(phif);
    fftw_free(gradf);
}
