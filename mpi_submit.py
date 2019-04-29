import sys, os
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse
import pickle
import importlib
import subprocess
import numpy as np
from mpi4py import MPI
import h5py
import time
from pydoc import locate
from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm
from pyuoi.mpi_utils import Bcast_from_root, Gatherv_rows
from utils import gen_data
from utils import FNR, FPR, selection_accuracy, estimation_error

total_start = time.time()

###### Command line arguments #######
parser = argparse.ArgumentParser()

parser.add_argument('arg_file')
parser.add_argument('results_file', default = 'results.h5')
parser.add_argument('exp_type', default = 'UoILasso')
args = parser.parse_args()
#######################################

exp_type = args.exp_type
results_file = args.results_file

# Create an MPI comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()    
   
if exp_type in ['UoILasso', 'UoIElasticNet', 'GTV']:
    partype = 'uoi'
else:
    partype = 'reps'

# Open the arg file and read out the number of total_tasks and n_features
f = open(args.arg_file, 'rb')
total_tasks = pickle.load(f)
n_features = pickle.load(f)

if partype == 'reps':
    # Chunk up iter_param_list to distribute across iterations
    chunk_param_list = np.array_split(np.arange(total_tasks), numproc)
    chunk_idx = rank
    num_tasks = len(chunk_param_list[chunk_idx])
else:
    chunk_idx = 0
    num_tasks = total_tasks

# Initialize arrays to store data in. Assumes that n_features
# is held constant across all iterations
if (partype == 'uoi' and rank == 0) or partype == 'reps':
    beta_hats = np.zeros((num_tasks, n_features))

    # result arrays: scores
    fn_results = np.zeros(num_tasks)
    fp_results = np.zeros(num_tasks)
    r2_results = np.zeros(num_tasks)
    r2_true_results = np.zeros(num_tasks)

    BIC_results = np.zeros(num_tasks)
    AIC_results = np.zeros(num_tasks)
    AICc_results = np.zeros(num_tasks)

    FNR_results = np.zeros(num_tasks)
    FPR_results = np.zeros(num_tasks)
    sa_results = np.zeros(num_tasks)
    ee_results = np.zeros(num_tasks)
    median_ee_results = np.zeros(num_tasks)
    
for i in range(num_tasks):
    start = time.time()
    
    params = pickle.load(f)
    sigma = params['sigma']
    beta = params['betas']
    seed = params['seed']
    # Generate data
    X, X_test, y, y_test = gen_data(params['n_samples'], params['n_features'],
                                    params['kappa'], sigma, beta, seed)

    exp = locate('exp_types.%s' % exp_type)
    model = exp.run(X, y, params)

    if (partype == 'uoi' and rank == 0) or partype == 'reps':
        #### Calculate and log results
        beta_hat = model.coef_.ravel()
        beta_hats[i, :] = beta_hat.ravel()
        fn_results[i] = np.count_nonzero(beta[beta_hat == 0, 0])
        fp_results[i] = np.count_nonzero(beta_hat[beta.ravel() == 0])
        r2_results[i] = r2_score(y_test, np.dot(X_test, beta_hat))
        r2_true_results[i] = r2_score(y_test, np.dot(X_test, beta))
    # Score functions have been modified, requiring us to first calculate log-likelihood
        llhood = log_likelihood_glm('normal', y_test, np.dot(X_test, beta))
        try:
            BIC_results[i] = BIC(llhood, np.count_nonzero(beta_hat), n_samples)
        except:
            BIC_results[i] = np.nan
        try:
            AIC_results[i] = AIC(llhood, np.count_nonzero(beta_hat))
        except:
            AIC_results[i] = np.nan
        try:
            AICc_results[i] = AICc(llhood, np.count_nonzero(beta_hat), n_samples)
        except:
            AICc_results[i] = np.nan
        # Perform calculation of FNR, FPR, selection accuracy, and estimation error
        # here:

        FNR_results[i] = FNR(beta.ravel(), beta_hat.ravel())
        FPR_results[i] = FPR(beta.ravel(), beta_hat.ravel())
        sa_results[i] = selection_accuracy(beta.ravel(), beta_hat.ravel())
        ee, median_ee = estimation_error(beta.ravel(), beta_hat.ravel())
                
        ee_results[i] = ee
        median_ee_results[i] = median_ee
    del params
    print('Process %d completed outer loop %d/%d' % (rank, i, num_tasks - 1))
    print(time.time() - start)
       
# Save results. If parallelizing over reps, concatenate all arrays together first
if partype == 'reps':
    fn_results = np.array(fn_results)
    print(fn_results.shape)
    fn_results = Gatherv_rows(fn_results, comm, root = 0)
    if rank == 0:
        print(fn_results.shape)

    fp_results = np.array(fp_results)
    fp_results = Gatherv_rows(fp_results, comm, root = 0)

    r2_results = np.array(r2_results)
    r2_results = Gatherv_rows(r2_results, comm, root = 0)

    r2_true_results = np.array(r2_true_results)
    r2_true_results = Gatherv_rows(r2_true_results, comm, root = 0)

    beta_hats = np.array(beta_hats)
    beta_hats = Gatherv_rows(beta_hats, comm, root = 0)

    BIC_results = np.array(BIC_results)
    BIC_results = Gatherv_rows(BIC_results, comm, root = 0)

    AIC_results = np.array(AIC_results)
    AIC_results = Gatherv_rows(AIC_results, comm, root = 0)

    AICc_results = np.array(AICc_results)
    AICc_results = Gatherv_rows(AICc_results, comm, root = 0)

    FNR_results = np.array(FNR_results)
    FNR_results = Gatherv_rows(FNR_results, comm, root = 0)

    FPR_results = np.array(FPR_results)
    FPR_results = Gatherv_rows(FPR_results, comm, root = 0)

    sa_results = np.array(sa_results)
    sa_results = Gatherv_rows(sa_results, comm, root = 0)

    ee_results = np.array(ee_results)
    ee_results = Gatherv_rows(ee_results, comm, root = 0)

    median_ee_results = np.array(median_ee_results)
    median_ee_results = Gatherv_rows(median_ee_results, comm, root = 0)

f.close()    
if rank == 0:
    # Save results
    with h5py.File(results_file, 'w') as results:

        results['fn'] = fn_results
        results['fp'] = fp_results
        results['r2'] = r2_results
        results['r2_true'] = r2_true_results
        results['beta_hats'] = beta_hats
        results['BIC'] = BIC_results
        results['AIC'] = AIC_results
        results['AICc'] = AICc_results

        results['FNR'] = FNR_results
        results['FPR'] = FPR_results
        results['sa'] = sa_results
        results['ee'] = ee_results
        results['median_ee'] = median_ee_results
    print('Total time: %f' % (time.time() - total_start))
    print('Job completed!')
    
