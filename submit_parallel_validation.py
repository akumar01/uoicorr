import argparse
import h5py
import numpy as np
from sklearn.model_selection import KFold
from job_manager import grab_files, run_
import itertools
import os
import pickle
from neuropacks.ecog import ECOG
from neuropacks.pvc11 import PVC11

def create_job_structure(datapath, jobdir, dataset, n_chunks = 10):

    exp_types = ['CV_Lasso', 'EN', 'UoILasso', 'UoIElasticNet', 'GTV']
    
    job_times = {'CV_Lasso' : '00:30:00', 'EN' : '0:30:00', 
                 'UoILasso' : '03:00:00', 'UoIElasticNet' : '08:00:00',
                 'GTV' : '24:00:00'}
    n_folds = 5

    # Create directory structure:
    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    # TO DO: Decide which form of the data to load in this file
    # Then, for each dataset, generate the folds in the proper way
    # and then save away the data, and generate all the necessary 
    # sbatch files. We have opted for shared queue as opposed to
    # trying to use MPI


    if not os.path.exists('%s/%s' % (jobdir, dataset)):
        os.makedirs('%s/%s' % (jobdir, dataset))

    if dataset == 'A1':

        ecog = ECOG(datapath)
        response_matrix = ecog.get_response_matrix(bounds = (40, 60),
                                                    band ='HG')
 
    elif dataset == 'PVC':

        pvc = PVC11(datapath)
        response_matrix = pvc.get_response_matrix(transform = 'square_root')

    elif 'MEG' in dataset:

        # Get the index
        idx = int(dataset.split('MEG')[1])

        meg = h5py.File(datapath, 'r')
        response_matrix = meg['data'][idx]
        
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
    chunk_columns = np.array_split(np.arange(n_nodes), n_chunks)

    # Write train/test splits of data to file
    data_files = []
    for i in range(n_folds):
        data_file = '%s/%s/dat_fold%d' % (jobdir, dataset, i)
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

                sbname = '%s/sbatch_fold%d_col%d.sh' % (sbatch_dir, i, j)
                jobname = '%s_%s_fold%d_col%d' % (dataset, exp_type,i, j)
                results_file = '%s/%s.dat' % (sbatch_dir, jobname)
                outfile = '%s/%s.o' % (sbatch_dir, jobname)
                errfile = '%s/%s.e' % (sbatch_dir, jobname)

                job_time = job_times[exp_type]

                arg_string = '%s %s %s -i ' % (data_files[i], results_file, exp_type)

                # Append the target column indices to the arg string
                for idx in chunk_columns[j]:
                    arg_string += '%d ' % idx
                                
                with open(sbname, 'w') as sb:
                    # Arguments common across jobs
                    sb.write('#!/bin/bash\n')
                    sb.write('#SBATCH -q shared\n')
                    sb.write('#SBATCH -t %s\n' % job_time)
                    sb.write('#SBATCH -n 1\n')
                    sb.write('#SBATCH -C haswell\n')
                    sb.write('#SBATCH --job-name=%s\n' % jobname)
                    sb.write('#SBATCH --out=%s\n' % outfile)
                    sb.write('#SBATCH --error=%s\n' % errfile)
                    sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
                    sb.write('#SBATCH --mail-type=FAIL\n')
                                        
                    # Work with out own Anaconda environment
                    sb.write('source ~/anaconda3/bin/activate\n')
                    sb.write('source activate nse\n')
                    sb.write('srun python3 -u ~/repos/uoicorr/batch_validation.py %s' 
                             % arg_string)

def run_jobs(jobdir, dataset, exp_type = None, size = None, run = False):

    run_files = grab_files('%s/%s' % (jobdir, dataset), '*.sh', exp_type)
    
    if size is not None:
        run_files = np.random.choice(run_files, replace = False, size = size)

    print('Submitting %d jobs' % len(run_files))
    if run:
        for run_file in run_files:
            run_(run_file)
