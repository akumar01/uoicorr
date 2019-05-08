import argparse
import h5py
import numpy as np
from sklearn.model_selection import KFold
from job_manager import grab_files, run_
import itertools
import os

def create_job_structure(datapath, jobdir, dataset, n_chunks = 10):

	n_folds = 5

	# Create directory structure:
	if not os.path.exists(jobdir):
		os.makedirs(jobdir)

	# TO DO: Decide which form of the data to load in this file
	# Then, for each dataset, generate the folds in the proper way
	# and then save away the data, and generate all the necessary 
	# sbatch files. We have opted for shared queue as opposed to
	# trying to use MPI

	if dataset == 'A1':
		from neuropacks.neuropacks import ECOG
		
		if not os.path.exists('%s/%s' % (jobdir, dataset)):
			os.makedirs('%s/%s' % (jobdir, dataset))

		ecog = ECOG(datapath)
		response_matrix = ecog.get_response_matrix(bounds = (40, 60),
													band ='HG')

		# Normalize
		response_matrix = response_matrix - np.mean(response_matrix, axis = 0)
		response_matrix = response_matrix/np.std(response_matrix, axis = 0)

		# Partition into folds
		kfold = KFold(n_splits = n_folds, shuffle = True, random_state = 25)
		train_data_folds = []
		test_data_folds = []
		for train_idxs, test_idxs in kfold.split(response_matrix):
			train_data_folds.append(response_matrix[train_idxs, :])
			test_data_folds.append(response_matrix[test_idxs, :])

		# Partition columns of coupling model fits
		n_nodes = response_matrix.shape[1]
		chunk_columns = np.array_split(np.arange(n_nodes), n_chunks = n_chunks)

	# Write train/test splits of data to file
	data_files = []
	for i in range(n_folds):
		data_file = '%s/%s/dat_fold%d_c%d' % (jobdir, dataset, j)
		with open(data_file, 'wb') as f:
			f.write(pickle.dumps(train_data_folds[i]))
			f.write(pickle.dumps(test_data_folds[i]))
		data_files.append(data_file)

	# Create sbatch files 
	for i in range(n_folds):
		for j in range(len(chunk_columns)):
			for exp_type in exp_types:

				sbatch_dir = '%s/%s/%s' % (jobdir, dataset, exp_type)

				if not os.path.exists(sbatch_dir):
					os.makedirs(sbatch_dir)

				sbname = 'sbatch_fold%d_col%d.sh' % (i, j)
				jobname = '%s_%s_fold%d_col%d' % (dataset, exp_type,i, j)
				results_file = '%s/%s.dat' % (sbatch_dir, jobname)
				outfile = '%s/%s.o' % (sbatch_dir, jobname)
				errfile = '%s/%s.e' % (sbatch_dir, jobname)

				job_time = ???

				arg_string = '%s %s %s -i' % (data_files[i], results_file, exp_type)

				# Append the target column indices to the arg string
				[arg_string += '%d' % idx for idx in chunk_columns[j]]

				with open(sbname, 'w') as sb:
		            # Arguments common across jobs
		            sb.write('#!/bin/bash\n')
		            sb.write('#SBATCH -q shared\n')
		            sb.write('#SBATCH --n_tasks 1\n')
		            sb.write('#SBATCH --cpus-per-task=4')
		            sb.write('#SBATCH -t %s\n' % job_time)

		            sb.write('#SBATCH --job-name=%s\n' % jobname)
		            sb.write('#SBATCH --out=%s/%s.o\n' % outfile)
		            sb.write('#SBATCH --error=%s/%s.e\n' % errfile)
		            sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
		            sb.write('#SBATCH --mail-type=FAIL\n')

		            # Work with out own Anaconda environment
		            sb.write('source ~/anaconda3/bin/activate\n')
		            sb.write('source activate nse\n')
		            sb.write('srun python3 -u ~/repos/uoicorr/batch_validation.py %s' 
		            		 % arg_string)

def run_jobs(jobdir, dataset, exp_type, size = None, run = False):

	run_files = grab_files(jobdir, '*.sh', '%s/%s' % (dataset, exp_type))
    
    if size is not None:
    	run_files = np.random.choice(run_files, replace = False, size = size)

    print('Submitting %d jobs' % len(run_files))

    if cont:
    	for run_file in run_files:
    		run_(run_file)