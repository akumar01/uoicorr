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

if 'const_beta' in list(args.keys()):
    const_beta = args['const_beta']
else:
    const_beta = False

# Type of parallelization. 'reps' breaks up the outer loop in this file, 
# whereas 'uoi' delegates to built-in uoi functionality
partype = args['partype']

# Keys that will be iterated over in the outer loop of this function
sub_iter_params = args['sub_iter_params']

# If any of the specified parameters are not in the list/arrays, wrap them
for key in sub_iter_params:
    if type(args['sub_iter_params']) != list:
        args['sub_iter_params'] = [args['sub_iter_params']]

# Complement of the sub_iter_params:
const_keys = list(set(args.keys()) - set(sub_iter_params))

const_args = {k: args[k] for k in const_keys}

# Combine reps and parameters to be cycled through into a single iterable. Store
# the indices pointing to the corresponding rep/parameter list for easy unpacking
# later on
iter_param_list = args['reps']\
                    * list(itertools.product(*[args[key] for key in args['sub_iter_params']]))
iter_idx_list = args['reps']\
                     * list(itertools.product(*[np.arange(len(args[key])) 
                                        for key in args['sub_iter_params']]))
# Specify type of covariance matrix and which
# fitting procedure to use
cov_type = args['cov_type']
exp_type = args['exp_type']

# Unpack other args
n_features = args['n_features']
block_size = args['block_size']
kappa = args['kappa']
est_score = args['est_score']
reps = args['reps']
sparsity = args['sparsity']
results_file = args['results_file']
betadist = args['betadist']
n_samples = args['n_samples']

# Determines the type of experiment to do 
# exp = importlib.import_module(exp_type, 'exp_types')
exp = locate('exp_types.%s' % exp_type)

# Create an MPI comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()

args['comm'] = comm

# Keep beta fixed across repetitions
if const_beta:
    if partype == 'reps':
        if rank == 0:
            beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)
        else:
            beta = None
        beta = Bcast_from_root(beta, comm, root = 0)
    else:
        beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)

# Chunk up iter_param_list to distribute across iterations
chunk_param_list = np.array_split(iter_param_list, numproc)

# Initialize arrays to store data in
if (partype == 'uoi' and rank == 0) or partype == 'reps':
    shape = int(len(iter_param_list)/numproc)
    betas = np.zeros((shape, n_features))
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


for i, iter_param in enumerate(iter_param_list)

    # Merge iter_param and constant_paramss
    params = {**iter_param, **const_args}

    # Generate new model coefficients for each repetition
    if not const_beta:
        beta = gen_beta(params['n_features'], params['block_size'],
                        params['sparsity'], betadist = params['betadist'])

    betas[i, :] = beta.ravel()

    if (partype == 'uoi' and rank == 0) or partype == 'reps'
        # Return covariance matrix
        # If the type of covariance is interpolate, then the matricies have been
        # pre-generated

        if cov_type == 'interpolate':
            sigma = np.array(cov_param['sigma'])
        else:
            sigma = gen_covariance(params['cov_type'], params['n_features'],
                                params['block_size'], **params['cov_param'])
        X, X_test, y, y_test = gen_data(n_samples = params['n_samples'], 
        n_features= params['n_features'], kappa = params['kappa'],
        covariance = sigma, beta = beta)
    else:
        X = None
        y = None
        sigma = None 

    if partype == 'uoi':
        X = Bcast_from_root(X, comm, root = 0)
        y = Bcast_from_root(y, comm, root = 0)
        sigma = Bcast_from_root(sigma, comm, root = 0)

    params['cov'] = sigma

    # Call to UoI
    model = exp.run(X, y, params)

    if (partype == 'uoi' and rank == 0) or partype == 'reps':
        #### Calculate and log results
        beta_hat = model[0].coef_.ravel()
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
        print(time.time() - start)

# Save results. If parallelizing over reps, concatenate all arrays together first
if partype == 'reps' and rank == 0:

    fn_results = Gatherv_rows(fn_results, comm, root = 0)
    fp_results = Gatherv_rows(fp_results, comm, root = 0)
    r2_results = Gatherv_rows(r2_results, comm, root = 0)
    r2_true_results = Gatherv_rows(r2_true_results, comm, root = 0)
    betas = Gatherv_rows(betas, comm, root = 0)
    beta_hats = Gatherv_rows(beta_hats, comm, root = 0)
    BIC_results = Gatherv_rows(BIC_results, comm, root = 0)
    AIC_results = Gatherv_rows(AIC_results, comm, root = 0)
    AICc_results = Gatherv_rows(AICc_results, comm, root = 0)

    FNR_results = Gatherv_rows(FNR_results, comm, root = 0)
    FPR_results = Gatherv_rows(FPR_results, comm, root = 0)
    sa_results = Gatherv_rows(sa_results, comm, root = 0)
    ee_results = Gatherv_rows(ee_results, comm, root = 0)
    median_ee_results = Gatherv_rows(median_ee_results, root = 0)

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
    print('Total time: %f' % (time.time() - total_start))