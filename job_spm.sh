#!/bin/bash
#SBATCH --partition=partMath
#SBATCH -N 1
#SBATCH --ntasks=32

# Load the MPI environment
export PATH=/beegfs/software/mpich/mpich-4.0.3/bin:$PATH

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beegfs/software/fftw/fftw-3.3.10/lib:/beegfs/software/mpich/mpich-4.0.3/lib:$PWD/lib

if [ ! -d "log/" ]; then
    mkdir log
fi

# Use the SLURM_NTASKS environment variable
/beegfs/software/mpich/mpich-4.0.3/bin/mpirun -np $SLURM_NTASKS ./spm_yzys > log/DIS_S1.log 2>&1

find . -name "*.out" |xargs rm -rfv
