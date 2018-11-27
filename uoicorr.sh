#!/bin/bash
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH --job-name=uoicorr
#SBATCH --out=/global/homes/s/sachdeva/uoicorr/uoicorr_o.txt
#SBATCH --error=/global/homes/s/sachdeva/uoicorr/uoicorr_e.txt
#SBATCH --mail-user=pratik.sachdeva@berkeley.edu
#SBATCH --mail-type=FAIL

module load python/3.6-anaconda-4.4
python3 /global/homes/s/sachdeva/uoicorr/uoicorr.py --results_file=/global/homes/s/sachdeva/uoicorr/uoicorr.h5
