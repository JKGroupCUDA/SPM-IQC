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

int main(int argc, char **argv)
{
	double start,end,start1,end1,start2,end2,start3,end3;
	char filename[100];
	int Ini_Phase = 1;

	//const ptrdiff_t N0=N, N1=N;	
	MPI_Init(&argc, &argv);
    	fftw_mpi_init();
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();	
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	/*-----------------Model parameters and computational-domain discretization--------------*/
	LPsystParameter();

	/*-------------------------------------------------------------------------------------------*/
	// Initialize data and allocate memory

    	memAllocation_all();

	/* k-space */
	get_kindex_all();

	MPI_Barrier(MPI_COMM_WORLD);
	/*-------------------------------------------------------------------------------------------*/
	
	
    	/* get local data size and allocate */
 	My_mpi_fft_set_up();	
    	/* initialize data to u_F_local,k_square_local */	
	memAllocation_local();
	
	//Ini_local_k_spase();		  // Allocate k_square_local, empck2_local, and ikt2_local
	Distribute_C(kindex,kindex_local);
	get_ksquare_local();
	get_epmckt2_local();
	get_ikt2_local();

	// Free global k-space data after it is no longer needed
	Memfree_global_kspace();
	//my_ifft(u_F_local,u_R_local);

	/*----------------------Load saddle point------------------------------------------------------*/
	MPI_Barrier(MPI_COMM_WORLD);
	start1 = MPI_Wtime();
	//get_min_u_F();
	sprintf(filename, "./result/%s","S1.txt");
	//sprintf(filename, "./result/%s","x0_new1.txt");
	if(myid == 0)
	{
		cout << "load: " << filename << endl; 
		load_phi(u_R,filename);	
	}
	MPI_Scatter(u_R,alloc_local,MPI_C_DOUBLE_COMPLEX,u_R_local,alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);	

	my_fft(u_R_local,u_F_local);
	
	MPI_Barrier(MPI_COMM_WORLD);	
	end1 = MPI_Wtime();

	/*------------------------------------compute ene  in myid-------------------------*/

	double E = 0.0;
	E = get_Ene();
	MPI_Bcast ( &E, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
	MPI_Barrier(MPI_COMM_WORLD);
	if(myid ==0)
		printf("\nEnergy of transition state : %.15e\n",E);

	/*---------------------------------------------------------------------------------*/

	MPI_Barrier(MPI_COMM_WORLD);
	start2 = MPI_Wtime();
	option1 = (Option *)malloc(sizeof(Option));
	hiosd_lobpcg_initialization(option1,3);

	// Check whether the V1_saddle.txt file exists
 	sprintf(filename, "./result/V1_saddle.txt");
	FILE *file = fopen(filename, "r");
 	if (file != NULL) 
	{
     		// If the file exists, close it and skip the LOBPCG calculation
        	fclose(file);
        	printf("File  exists. Skipping LOBPCG calculation.\n");
		load_Option_V(option1, filename);
	} 
	else 
	{		
        	printf("File does not exist. Performing LOBPCG calculation.\n");
		for(int i=0;i<20;i++)
		{
			hiosd_lobpcg_inip(u_R_local ,option1 );
		}
		Save_Option_V(option1,filename);
		cout << "save : " << filename << endl;

	}
	end2 = MPI_Wtime();	


	/*-----------------------------------gradient descent---------------------------------------*/
	start3 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	FuncsLinear2Cplx(u_R_local, CplxDofs_local,1.0, u_R_local,0.001, option1->V[0]);//u_R_local+v0	
	my_fft(u_R_local,u_F_local);
	get_min_u_F();// Compute the next structure
	//critical_point();// Compute the saddle point
	double res = 0.0;
	end3 = MPI_Wtime();

	E = get_Ene();
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);
	if(myid ==0)
		printf("\nEnergy of relaxed minimum ：%.15e\n",E);

	MPI_Gather(u_F_local,alloc_local,MPI_C_DOUBLE_COMPLEX,u_F,alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
	MPI_Gather(u_R_local,alloc_local,MPI_C_DOUBLE_COMPLEX,u_R,alloc_local,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
	
	/*---------------------------Save x_new -----------------------------------*/
	sprintf(filename, "./result/%s","Min.txt");
	if(myid == 0)
	{
		cout << "save: " << filename << endl;
		Save_phi(u_R,filename);
	}
	/*-----------------------------------------------------------------------*/


	MPI_Barrier(MPI_COMM_WORLD);
	//hiosd_lobpcg_initialization(option1,4);
	for(int i=0;i<25;i++)
	{
		hiosd_lobpcg_inip(u_R_local ,option1 );
	}
	
	/*-------------------------------------------Save V--------------------------------------------*/
	sprintf(filename, "./result/%s","V0_Min.txt");
	Save_Option_V(option1,filename);


	//MPI_Barrier(MPI_COMM_WORLD);		
	//memReleaser();
	fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan1);

	Memfree_global_data();
	Memfree_local_kspace();
	Memfree_local_data();
	
	Memfree_Option(option1);


	MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	end = MPI_Wtime();

	if (myid == 0) 
	{
		printf("Running time of the whole program %f s \n", end-start);
		printf("Running time of the calculate  minimum point program %f s \n", end3-start3);
		printf("Running time of the LOBPCG program %f s \n", end2-start2);
		//printf("Running time of the CSD program %f s \n", end3-start3);
	}

    	MPI_Finalize();

}
