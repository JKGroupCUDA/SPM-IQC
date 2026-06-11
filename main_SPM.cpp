#include <fftw3-mpi.h>
#include <math.h>
#include "Data.h"
#include "Initialization.h"
#include "LPsrc.h"
#include "BasicOperators.h"
#include "LOBPCG.h"
#include "HISD.h"
#include "GOSD.h"
#include "File_Op.h"
#include "My_FFT.h"
#include "Memfree.h"
#include "SPM.h"

int main(int argc, char **argv)
{
	double start,end,start1,end1,start2,end2,start3,end3;
	char filename[100];

	MPI_Init(&argc, &argv);
    	fftw_mpi_init();
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();	
	
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	/*-----------------Model parameters and computational-domain discretization--------------*/
	LPsystParameter();    
	/*--------------------------------------------------------------------------------*/
	// Initialize data and allocate memory
    	memAllocation_all();

	/* k-space */
	get_kindex_all();
	
	/*   Select initial value   */
	get_initial_value(u_F,Ini_Phase,0.0);	

	MPI_Barrier(MPI_COMM_WORLD);
	/*--------------------------------------------------------------------------------*/
	
	
    /* get local data size and allocate */
    /* create plan for in-place forward DFT */
	My_mpi_fft_set_up();
	
    /* initialize data to u_F_local,k_square_local */
	memAllocation_local();
	//Distribute_C(u_F, u_F_local); 
	MPI_Scatter(u_F,alloc_local,MPI_C_DOUBLE_COMPLEX,u_F_local,alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);

	//Ini_local_kspase();		  // Allocate k_square_local, empck2_local, and ikt2_local
	Distribute_C(kindex,kindex_local);
	get_ksquare_local();
	get_epmckt2_local();
	get_ikt2_local();

	// Free global k-space data after it is no longer needed
	Memfree_global_kspace();

	my_ifft(u_F_local,u_R_local);

/*----------------------Compute the minimum state--------------------------------------------------*/
	MPI_Barrier(MPI_COMM_WORLD);
	start1 = MPI_Wtime();
	get_min_u_F();
	MPI_Barrier(MPI_COMM_WORLD);	
	end1 = MPI_Wtime();

/*-----------compute ene  in myid--------------------------------------------------*/
	
	my_fft(u_R_local,u_F_local);
	double E = 0.0;
	E = get_Ene();
	MPI_Bcast ( &E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
	MPI_Barrier(MPI_COMM_WORLD);
/*---------------------------------------------------------------------------------*/

	MPI_Barrier(MPI_COMM_WORLD);
	start2 = MPI_Wtime();
	option1 = (Option *)malloc(sizeof(Option));
	hiosd_lobpcg_initialization(option1,2);
	if(myid == 0)
	{
		printf("\n--------------Calculating eigenvalues--------------------------\n");
	}
	for(int i=0;i<10;i++)
	{
		hiosd_lobpcg_inip(u_R_local ,option1 );
	}
	end2 = MPI_Wtime();

	/*---------------------------------Save V------------------------------------------*/
	//sprintf(filename, "./result/%s","V0.txt");
	//Save_Option_V(option1,filename);
	//load_Option_V(option1, filename);

	if(myid ==0)
		printf("\nEnergy of metastable state :  %.15e\n",E);




/*-----------------------------------SPM-----------------------------*/
	
	
	printf("\n =========================SPM================================= \n ");
	for(int i=0;i<CplxDofs_local;i++)
	{
		Phi0[0][i][0] = u_R_local[i][0];
		Phi0[0][i][1] = 0.0;

		// This perturbation uses the first LOBPCG eigenvector by default.
		// If the LOBPCG iterations above are disabled, option1->V[0]
		// remains the random direction generated during initialization.
		Phi0[1][i][0] = u_R_local[i][0] + 0.001*option1->V[0][i][0];		
		Phi0[1][i][1] =  0.0;
	}
	

	Spring_Pairing(Phi0, 30000, 1e-9);
	Cpy(Phi0[1],u_R_local,CplxDofs_local);
	
	my_fft(u_R_local,u_F_local);
	double Es;
	Es = get_Ene();
	MPI_Barrier(MPI_COMM_WORLD);
	if(myid ==0)
		printf("\nEnergy of transition state ： %.15e\n",Es);


	MPI_Gather(u_R_local,alloc_local,MPI_C_DOUBLE_COMPLEX,u_R,alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);	
	sprintf(filename, "./result/%s","S1.txt");
	if(myid == 0)
	{
		cout << "save saddle : "<< filename << endl;
		Save_phi(u_R,filename);
	}
	/*-----------------------------------------------------------------------*/


	fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);

	Memfree_global_data();
	Memfree_local_kspace();
	Memfree_local_data();

	Memfree_Option(option1);
	//Memfree_Option(option2);

	MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	end = MPI_Wtime();

	if (myid == 0) 
	{
		printf("\nRunning time of the whole program %f s \n", end-start);

	}

    MPI_Finalize();
}
