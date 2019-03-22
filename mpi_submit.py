import sys, os
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse
import json
import importlib
import itertools
import subprocess
import numpy as np
from mpi4py import MPI
import h5py
import time
from pydoc import locate
from scipy.linalg import block_diag
from sklearn.metrics import r2_score

# Need to add pyuoi to path
parent_path, current_dir = os.path.split(os.path.abspath('.'))

# Crawl up to the repos folder
while current_dir not in ['repos']:
    parent_path, current_dir = os.path.split(parent_path)

p = os.path.join(parent_path, current_dir)

# Add uoicorr and pyuoi to the path
if '%s/uoicor' % p not in sys.path:
    sys.path.append('%s/uoicorr' % p)
if '%s/PyUoI' % p not in sys.path:
    sys.path.append('%s/PyUoI' % p)

from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm
from pyuoi.mpi_utils import Bcast_from_root

from utils import gen_beta, gen_data, gen_covariance
from utils import FNR, FPR, selection_accuracy, estimation_error

if __name__ == '__main__':
    total_start = time.time()
    ###### Command line arguments #######
    parser = argparse.ArgumentParser()
    
    # Param file from which to create job scripts
    parser.add_argument('arg_file', default=None)
    args = parser.parse_args()
    #######################################


    # Load param file
    with open(args.arg_file, 'r') as f: 
        args = json.load(f)
        f.close()

    ## Initialize a bunch of stuff ###
    # Unpack args
    n_features = args['n_features']
    block_size = args['block_size']
    kappa = args['kappa']
    est_score = args['est_score']
    reps = args['reps']
    sparsity = args['sparsity']
    results_file = args['results_file']
    betadist = args['betadist']
    n_samples = args['n_samples']

    if 'const_beta' in list(args.keys()):
        const_beta = args['const_beta']
    else:
        const_beta = False

    # Specify type of covariance matrix and which
    # fitting procedure to use
    cov_type = args['cov_type']
    cov_params = args['cov_params']
    exp_type = args['exp_type']

    # Wrap dictionary in a list
    if type(cov_params) != list:
        cov_params = [cov_params]

    # Determines the type of experiment to do 
    # exp = importlib.import_module(exp_type, 'exp_types')
    exp = locate('exp_types.%s' % exp_type)

    # Use the n_models flags to allow experiments to return
    # multiple models over multiple parameters
    shape = (reps, len(cov_params))

    betas = np.zeros((reps, n_features))
    beta_hats = np.zeros(shape + (n_features,))

    # result arrays: scores
    fn_results = np.zeros(shape)
    fp_results = np.zeros(shape)
    r2_results = np.zeros(shape)
    r2_true_results = np.zeros(shape)

    BIC_results = np.zeros(shape)
    AIC_results = np.zeros(shape)
    AICc_results = np.zeros(shape)

    FNR_results = np.zeros(shape)
    FPR_results = np.zeros(shape)
    sa_results = np.zeros(shape)
    ee_results = np.zeros(shape)
    median_ee_results = np.zeros(shape)

    # Create an MPI comm object
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Distribute reps and distinct cov_params (should really be any iterable parameter the user desires)

    # Keep beta fixed across repetitions. In this case, need to distribute beta across all mpi processes
    if const_beta:
        if rank == 0:
                beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)
        else:
            beta = None
        beta = Bcast_from_root(beta, comm, root = 0)
    else:
        beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)
    # Merge cov params into the desired number of repetitions
    rep_cov_params = itertools.repeat(cov_params, reps)

    # Chunk into the number of processes
    chunk_params = np.array_split(np.fromiter(rep_cov_params))

#    betas[rep, :] = beta.ravel()

    start = time.time()

    # Return covariance matrix
    # If the type of covariance is interpolate, then the matricies have been
    # pre-generated
    for i, cov_param in enumerate(chunk_params):

        if cov_type == 'interpolate':
            sigma = np.array(cov_param['sigma'])
        else:

        sigma = gen_covariance(cov_type, n_features, block_size, **cov_param)
        X, X_test, y, y_test = gen_data(n_samples = n_samples, 
        n_features= n_features, kappa = kappa, covariance = sigma, beta = beta)
        args['cov'] = sigma

        # Call to UoI
        model = exp.run(X, y, args)

    print('Total time: %f' % (time.time() - total_start))


        if rank == 0:
            #### Calculate and log results
            beta_hat = model[0].coef_.ravel()
            beta_hats[rep, cov_idx, :] = beta_hat.ravel()
            fn_results[rep, cov_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
            fp_results[rep, cov_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
            r2_results[rep, cov_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
            r2_true_results[rep, cov_idx] = r2_score(y_test, np.dot(X_test, beta))
            # Score functions have been modified, requiring us to first calculate log-likelihood
            llhood = log_likelihood_glm('normal', y_test, np.dot(X_test, beta))
            try:
                BIC_results[rep, cov_idx] = BIC(llhood, np.count_nonzero(beta_hat), n_samples)
            except:
                BIC_results[rep, cov_idx] = np.nan
            try:
                AIC_results[rep, cov_idx] = AIC(llhood, np.count_nonzero(beta_hat))
            except:
                AIC_results[rep, cov_idx] = np.nan
            try:
                AICc_results[rep, cov_idx] = AICc(llhood, np.count_nonzero(beta_hat), n_samples)
            except:
                AICc_results[rep, cov_idx] = np.nan

            # Perform calculation of FNR, FPR, selection accuracy, and estimation error
            # here:

            FNR_results[rep, cov_idx] = FNR(beta.ravel(), beta_hat.ravel())
            FPR_results[rep, cov_idx] = FPR(beta.ravel(), beta_hat.ravel())
            sa_results[rep, cov_idx] = selection_accuracy(beta.ravel(), beta_hat.ravel())
            ee, median_ee = estimation_error(beta.ravel(), beta_hat.ravel())
                    
            ee_results[rep, cov_idx] = ee
            median_ee_results[rep, cov_idx] = median_ee
            print(time.time() - start)

    if rank == 0:
        # Save results
        with h5py.File(results_file, 'w') as results:

            results['fn'] = fn_results
            results['fp'] = fp_results
            results['r2'] = r2_results
            results['r2_true'] = r2_true_results
            results['betas'] = betas
            results['beta_hats'] = beta_hats
            results['BIC'] = BIC_results
            results['AIC'] = AIC_results
            results['AICc'] = AICc_results

            results['FNR'] = FNR_results
            results['FPR'] = FPR_results
            results['sa'] = sa_results
            results['ee'] = ee_results
            results['median_ee'] = median_ee_results
