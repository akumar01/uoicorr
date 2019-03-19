#!/bin/bash
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -J n1
#SBATCH --out=n1.o
#SBATCH --error=n1.e
#SBATCH --mail-user=ankit_kumar@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00
#SBATCH -C haswell

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

source activate nse
srun -n 1 -c 1 --cpu_bind=threads python mpi_submit.py job0_params.json
