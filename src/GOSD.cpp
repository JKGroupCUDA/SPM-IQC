#include <fftw3-mpi.h>
#include "Data.h"
#include "LPsrc.h"
#include <math.h>
#include "BasicOperators.h"
#include "LOBPCG.h"
#include "HISD.h"
#include "GOSD.h"
#include "Initialization.h"
#include <algorithm>
using namespace std;

void critical_point()
{
    int k0 = 3;// Dimension of the null subspace
    hiosd_lobpcg_initialization(option2,1);
    option2->dt = 0.8;
    option2->betat = 8.0;
    option2->betau = 1.3;
    option2->epsf = 1e-9;
    option2->maxiter = 40000;
    option2->outputp = 100;
    option2->outputd = 1000;

  	FuncsLinear2Cplx(u_R_local, CplxDofs_local,1.0, u_R_local,-0.1, option1->V[k0]);
    //copy_option_V(option1->V[4],option2,0);
    
	for(int i=0;i<CplxDofs_local;i++)
    {
        option2->V[0][i][0] = option1->V[k0][i][0];
        option2->V[0][i][1] = 0.0;
    }
    
	fftw_complex **V0;
    V0 = (fftw_complex **)malloc(sizeof(fftw_complex*) * k0);
    for(int i=0;i<k0;i++)
    {
    	V0[i] = (fftw_complex *)malloc(sizeof(fftw_complex) * CplxDofs_local);
    }

    for(int i=0;i<k0;i++)
    {
    	for(int j=0;j<CplxDofs_local;j++)
    	{
    	    V0[i][j][0] = option1->V[i][j][0];
    	    V0[i][j][1] = 0.0;
    	}
    }
    hiosd_lobpcg_sieh_V0(u_R_local,option2,V0,k0);

    for(int i=0;i<k0;i++)
    {
    	fftw_free(V0[i]);
    }
    fftw_free(V0);
}

void critical_point_v()
{
    int k0 = 3;// Dimension of the null subspace
    hiosd_lobpcg_initialization(option2,1);
    option2->dt = 0.8;
    option2->betat = 8.0;
    option2->betau = 0.4;
    option2->epsf = 1e-7;
    option2->maxiter = 20000;
    option2->outputp = 100;
    option2->outputd = 1000;

	/*----------------------------------------v-----------------------------------------------*/
	fftw_complex *v,*v_local;
	v = (fftw_complex *)malloc(sizeof(fftw_complex) * CplxDofs);
	v_local =(fftw_complex *)malloc( sizeof(fftw_complex) * CplxDofs_local );

	get_initial_value(v,3,0.0);//hex
	Distribute_C(v, v_local);
	fftw_free(v);

	Cpy(v_local,data,CplxDofs_local);
	fftw_execute(plan2);
	Cpy(data,v_local,CplxDofs_local);
	/*------------------------------------------------------------------------------------------*/
	
	/*----------------------------------uR+v----------------------------------------------------*/
    FuncsLinear2Cplx(u_R_local, CplxDofs_local,1.0, u_R_local,0.1, v_local);
    //copy_option_V(option1->V[4],option2,0);
    
	for(int i=0;i<CplxDofs_local;i++)
    {
        option2->V[0][i][0] = v_local[i][0];
        option2->V[0][i][1] = 0.0;
    } 
	fftw_free(v_local);

	/*--------------------------------V0---------------------------------------------------------*/
	fftw_complex **V0;
    V0 = (fftw_complex **)malloc(sizeof(fftw_complex*) * k0);
    for(int i=0;i<k0;i++)
    {
    	V0[i] = (fftw_complex *)malloc(sizeof(fftw_complex) * CplxDofs_local);
    }

    for(int i=0;i<k0;i++)
    {
    	for(int j=0;j<CplxDofs_local;j++)
    	{
    	    V0[i][j][0] = option1->V[i][j][0];
    	    V0[i][j][1] = 0.0;
    	}
    }
	/*--------------------------------------------------------------------------------------------*/

    hiosd_lobpcg_sieh_V0(u_R_local,option2,V0,k0);

    for(int i=0;i<k0;i++)
    {
    	fftw_free(V0[i]);
    }
    fftw_free(V0);
}


void hiosd_lobpcg_sieh_V0(fftw_complex *x, Option *option,fftw_complex **V0,int k0)
{
    int k,maxiter,outputd,outputp;
    k = option->k;
    maxiter = option->maxiter;
    outputp = option->outputp;
    outputd = option->outputd;

    double dt,epsf,betat,betau;
    dt = option->dt;
    epsf = option->epsf;
    betat = option->betat;
    betau = option->betau;

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

    //printf("\n rayleighq(x,v1) = %.3e ",rayleighq(x,V[0]) );
    //printf("\n rayleighq(x,v2) = %.3e ",rayleighq(x,V[1]) );
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
    double nrmW ,temp1,temp1_all, nrmVpi;

    // Orthogonalize V
    for(int i=0;i<k;i++)
    {
        Orthoi_Vi(V,i);
        temp1 = nrmp(V[i]);
        FFT_dot_constant(V[i], CplxDofs_local,1.0/temp1);
    }

    fftw_complex *rhoCplx1,*f,*g,*gp,*Dx,*tmpp,*tmppF,*nlnF,*nlnR,*xp,*Dg;
	rhoCplx1 = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    f = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    gp = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    Dx = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    tmpp = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    tmppF = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    nlnR = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    nlnF = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    xp = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    g = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    Dg = (fftw_complex *)malloc(sizeof(fftw_complex)*CplxDofs_local);
    setCplxZero_v(tmpp,CplxDofs_local);
    setCplxZero_v(f,CplxDofs_local);
    setCplxZero_v(Dx,CplxDofs_local);
    setCplxZero_v(tmppF,CplxDofs_local);
    setCplxZero_v(nlnR,CplxDofs_local);
    setCplxZero_v(nlnF,CplxDofs_local);
    setCplxZero_v(xp,CplxDofs_local);
    setCplxZero_v(g,CplxDofs_local);
    setCplxZero_v(Dg,CplxDofs_local);

    ngrad_cam(x, f);// Compute the negative gradient
    setCplxZero_v(gp,CplxDofs_local);
    PV_Cplx(f,V,tmpp,k);// Verify correctness
    FuncsLinear2Cplx(Dx,CplxDofs_local,1.0,f,-2.0,tmpp);
    FuncRealAddAConst(Dx,CplxDofs_local,dt);

	//printf("\n tmpp: %.3e \n",nrmp(tmpp));
    double bta = 0.1,mean,mean_all,res,maxf,minf;
    while(iter < maxiter)
    {
        for(int i=0;i<3*k;i++) emp[i] = 1;
        PV_Cplx(f,V,tmpp,k);
        FFT_dot_constant(tmpp,CplxDofs_local,2.0);//tmpp=2*PV(f)

		//if(myid ==0)
		//	printf("tmpp: %.3e ",nrmp(tmpp));
        //g = f-tmpp
        FuncsLinear2Cplx(g,CplxDofs_local,1.0,f,-1.0,tmpp);
        //Dg = g-gp
        FuncsLinear2Cplx(Dg,CplxDofs_local,1.0,g,-1.0,gp);
        //gp = g
        FuncsLinear1Cplx(gp,CplxDofs_local,1.0,g);
        bta = abs(inpfp(Dx,Dg)/inpfp(Dg,Dg));

		//if(myid ==0 )
		//	printf("\n bta  %.3e  \n",bta );
        bta = min(bta,betat*dt);
        bta = max(bta,betau*dt);
		//if(myid ==0)
		//	printf("\n bta  %.3e  \n",bta );

        FuncsLinear1Cplx(xp,CplxDofs_local,1.0,x);//xp=x
        for(int i=0;i<CplxDofs_local;i++)
        {
            u2_R_local[i][0] = x[i][0]*x[i][0];
            u2_R_local[i][1] = 0.0;

            u3_R_local[i][0] = u2_R_local[i][0]*x[i][0];
            u3_R_local[i][1] = 0.0;
        }

		Cpy(x,data,CplxDofs_local);
		fftw_execute(plan1);
		norm_fft(data,CplxDofs_local);
		Cpy(data,rhoCplx1,CplxDofs_local);

        //nln = PARA_tau +  PARA_gamma*quadTerm - cubTerm;
        for(int i=0;i<CplxDofs_local;i++)
        {
			nlnR[i][0] = Tau*x[i][0] + Gamma*u2_R_local[i][0] - u3_R_local[i][0];
        	nlnR[i][1] = 0.0;
        }
        mean = mean_Cplx(nlnR, CplxDofs_local);// Requires communication
		MPI_Allreduce(&mean,&mean_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		mean_all = mean_all/numprocs;
		//if(myid ==0)
		//	printf(" mean_all : %.3e",mean_all);
        FuncRealAddAConst(nlnR,CplxDofs_local,-mean_all); //xx

		Cpy(nlnR,data,CplxDofs_local);
		fftw_execute(plan1);
		norm_fft(data,CplxDofs_local);
		Cpy(data,nlnF,CplxDofs_local);

		Cpy(tmpp,data,CplxDofs_local);
		fftw_execute(plan1);
		norm_fft(data,CplxDofs_local);
		Cpy(data,tmppF,CplxDofs_local);



	//	p = fftw_plan_dft(DimCpt,NCpt,nlnR,nlnF,FFTW_FORWARD, FFTW_ESTIMATE);
    //    fftw_execute(p);
    //    FFT_dot_constant(nlnF,CplxDofs_local,1.0/CplxDofs_local);

    //    p = fftw_plan_dft(DimCpt,NCpt,tmpp,tmppF,FFTW_FORWARD, FFTW_ESTIMATE);
    //    fftw_execute(p);
    //    FFT_dot_constant(tmppF,CplxDofs_local,1.0/CplxDofs_local);

        for(int i=0;i<CplxDofs_local;i++)
        {
            rhoCplx1[i][0]=( rhoCplx1[i][0]+bta*nlnF[i][0] - bta*tmppF[i][0])/(1+bta*epmckt2_local[i]);
            rhoCplx1[i][1]=0.0;
        }
		if(myid == 0)
			setCplxZero(rhoCplx1[0]);


		Cpy(rhoCplx1,data,CplxDofs_local);
		fftw_execute(plan2);
		Cpy(data,x,CplxDofs_local);
        //q = fftw_plan_dft(DimCpt,NCpt,rhoCplx1,x,FFTW_BACKWARD,FFTW_ESTIMATE);
        //fftw_execute(q);

        //printf("\n nrmp(x) %.3e \n",nrmp(x));
        for(int i=0;i<CplxDofs_local;i++)
        {
            Dx[i][0] = x[i][0] - xp[i][0];
            Dx[i][1] = 0.0;
        }


		/*---------------------V-----------------------------------------*/
        for(int i=0;i<k;i++)
        {
            hv_cam(x, V[i], U[i]);
            alpha[i] = inpfp(V[i],U[i]);
            FuncsLinear2Cplx(W[i], CplxDofs_local,1, U[i],-alpha[i],V[i]);
            precond_camnew(W[i]);
            Orthoi_v2V(W[i],V,k);
			Orthoi_v2V(W[i],V0,k0);
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
			Orthoi_v2V(Vp[i],V0,k0);
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
        //cout << "\n UU(0,0) : \n"<<UUm(0,0)<<endl<<endl;
        //cout<< "\n max UU  : \n"<<UUm.maxCoeff()<<endl<<endl;
        //cout<< "\n min UU  : \n"<<UUm.minCoeff()<<endl<<endl;
	    //cout<<"\n UU : \n" << UUm << endl<< endl;
        Pn = UUm*YYm.transpose();

		// Communication is required to sum Pn
		//Pn_all = 1.0*Pn;
		MPI_Barrier(MPI_COMM_WORLD);

		//MPI_Allreduce(&(Pn(0,0)),&(Pn_all(0,0)),m*m,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		// The communication below is faster
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
        MatrixXd vector1 = eta;
        vector1 = sort_eig(eta,value,m);
        sort(value.data(),value.data()+value.size());

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
			if(myid == 0)
				cout <<"\n"<< iter << " "<< value.transpose().leftCols(k)<<endl<<endl;
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
			Orthoi_v2V(V[i],V0,k0);
            temp1 = nrmp(V[i]);
            FFT_dot_constant(V[i], CplxDofs_local,1.0/temp1);
        }

        ngrad_cam(x,f);
        res = nrmp(f);

        if(iter % 10 == 0)
        {
			if(myid == 0)
				printf("\n %d \t bta = %.3e,  res = %.3e \n",iter,bta, res);
        }

        iter = iter +1;
		MPI_Barrier(MPI_COMM_WORLD);
        if(res < epsf)
        {
            break;
        }
    }

	// Update u_R_local and u_F_local once
	Cpy(x,u_R_local,CplxDofs_local);
	Cpy(rhoCplx1,u_F_local,CplxDofs_local);

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

	fftw_free(rhoCplx1);
    fftw_free(f);
    fftw_free(g);
    fftw_free(gp);
    fftw_free(Dx);
    fftw_free(tmpp);
    fftw_free(tmppF);
    fftw_free(xp);
    fftw_free(nlnF);
    fftw_free(nlnR);
    fftw_free(Dg);
}
