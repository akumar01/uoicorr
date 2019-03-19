#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J hello
#SBATCH --out=n2.o
#SBATCH --mail-user=ankit_kumar@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 1 --cpu_bind=threads python mpi_submit.py job0_params.json