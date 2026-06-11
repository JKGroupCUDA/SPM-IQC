#include <fftw3-mpi.h>
#include "My_FFT.h"

void my_fft(fftw_complex *x_R_local, fftw_complex *x_F_local)
{
	Cpy(x_R_local,data,CplxDofs_local);
	fftw_execute(plan1);
	Cpy(data,x_F_local,CplxDofs_local);
	for(int i=0;i<CplxDofs_local;i++)
	{
		x_F_local[i][0] = x_F_local[i][0]/CplxDofs;
		x_F_local[i][1] = x_F_local[i][1]/CplxDofs;
	}
}

void my_ifft(fftw_complex *x_F_local, fftw_complex *x_R_local)
{
	Cpy(x_F_local,data,CplxDofs_local);
	fftw_execute(plan2);
	Cpy(data,x_R_local,CplxDofs_local);	
}

//void My_mpi_fft_set_up()
//{
//	/* get local data size and allocate */
//	alloc_local = fftw_mpi_local_size_3d(N, N, N, MPI_COMM_WORLD,
//			&local_n0, &local_0_start);
//	data = fftw_alloc_complex(alloc_local);
//	CplxDofs_local = local_n0*N*N;
//	/* create plan for in-place forward DFT */
//	plan1 = fftw_mpi_plan_dft_3d(N, N, N, data, data,
//			MPI_COMM_WORLD,FFTW_FORWARD, FFTW_PATIENT);
//	plan2 = fftw_mpi_plan_dft_3d(N, N, N, data, data,
//			MPI_COMM_WORLD,FFTW_BACKWARD, FFTW_PATIENT);
//
//}




void My_mpi_fft_set_up()
{
    int rank;
    int new_plan_created = 0;
    char wisdom_filename[40];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create a wisdom filename that includes N */
    snprintf(wisdom_filename, sizeof(wisdom_filename), "fftw_wisdom_3d_N%d.dat", N);

    /* Get the local data size and allocate memory */
    alloc_local = fftw_mpi_local_size_3d(N, N, N, MPI_COMM_WORLD,
                    &local_n0, &local_0_start);
    data = fftw_alloc_complex(alloc_local);
    CplxDofs_local = local_n0 * N * N;

    /* Import wisdom on rank 0 */
    if (rank == 0) {
        if (!fftw_import_wisdom_from_filename(wisdom_filename)) {
            printf("Failed to import wisdom for N=%d, will create new plans\n", N);
        } else {
            printf("Successfully imported wisdom for N=%d\n", N);
        }
    }

    /* Broadcast wisdom to all processes */
    fftw_mpi_broadcast_wisdom(MPI_COMM_WORLD);

    /* Create forward and backward DFT plans using wisdom */
    plan1 = fftw_mpi_plan_dft_3d(N, N, N, data, data,
                    MPI_COMM_WORLD, FFTW_FORWARD, FFTW_WISDOM_ONLY | FFTW_PATIENT);
    plan2 = fftw_mpi_plan_dft_3d(N, N, N, data, data,
                    MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_WISDOM_ONLY | FFTW_PATIENT);

    /* Create new plans if suitable wisdom is not found */
    if (plan1 == NULL) {
        printf("Rank %d: No wisdom found for forward plan (N=%d), creating new plan\n", rank, N);
        plan1 = fftw_mpi_plan_dft_3d(N, N, N, data, data,
                    MPI_COMM_WORLD, FFTW_FORWARD, FFTW_PATIENT);
        new_plan_created = 1;
    }
    if (plan2 == NULL) {
        printf("Rank %d: No wisdom found for backward plan (N=%d), creating new plan\n", rank, N);
        plan2 = fftw_mpi_plan_dft_3d(N, N, N, data, data,
                    MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_PATIENT);
        new_plan_created = 1;
    }

    /* Export new wisdom on rank 0 only if new plans were created */
    MPI_Allreduce(MPI_IN_PLACE, &new_plan_created, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (new_plan_created && rank == 0) {
        if (!fftw_export_wisdom_to_filename(wisdom_filename)) {
            printf("Failed to export wisdom for N=%d\n", N);
        } else {
            printf("New wisdom for N=%d exported successfully\n", N);
        }
    }
}
