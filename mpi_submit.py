import sys, os
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse
import pickle
import struct
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
from mpi_utils.ndarray import Bcast_from_root, Gatherv_rows, Gather_ndlist
from utils import gen_data
from utils import FNR, FPR, selection_accuracy, estimation_error

import risk

total_start = time.time()

###### Command line arguments #######
parser = argparse.ArgumentParser()

parser.add_argument('arg_file')
parser.add_argument('results_file', default = 'results.h5')
parser.add_argument('exp_type', default = 'UoILasso')
parser.add_argument('--comm_splits', type=int, default = None)
parser.add_argument('-t', '--test', action = 'store_true')
args = parser.parse_args()
#######################################

exp_type = args.exp_type
results_file = args.results_file

# Create an MPI comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()

# If specified, split the comm object into subcommunicators. The number of splits will
# determine the number of parallel executions of the outer loop. This is useful for UoI
# to parallelize over both bootstraps and repetitions

if args.comm_splits is None:
    if exp_type in ['UoILasso', 'UoIElasticNet']:
        args.comm_splits = 1
    else:
        args.comm_splits = numproc

# Use array split to do comm.split
ranks = np.arange(numproc)
split_ranks = np.array_split(ranks, args.comm_splits)
color = [i for i in np.arange(args.comm_splits) if rank in split_ranks[i]][0]
subcomm_roots = [split_ranks[i][0] for i in np.arange(args.comm_splits)]

subcomm = comm.Split(color, rank)

rank = color
nchunks = args.comm_splits
subrank = subcomm.rank
numproc = subcomm.Get_size()

# Create a group including the root of each subcomm.
global_group = comm.Get_group()
root_group = MPI.Group.Incl(global_group, subcomm_roots)
roots_comm = comm.Create(root_group)

# Open the arg file and read out the index array, number of
# total_tasks and n_features


f = open(args.arg_file, 'rb')
index_loc = f.read(8)
index_loc = struct.unpack('L', index_loc)[0]
total_tasks = pickle.load(f)
n_features = pickle.load(f)
f.seek(index_loc, 0)
index = pickle.load(f)

# Chunk up iter_param_list to distribute across iterations
chunk_param_list = np.array_split(np.arange(total_tasks), nchunks)
chunk_idx = rank
num_tasks = len(chunk_param_list[chunk_idx])

print('rank: %d, subrank: %d, color: %d' % (comm.rank, subrank, color))
print('num_tasks: %d' % num_tasks)

# Initialize arrays to store data in. Assumes that n_features
# is held constant across all iterations
if subrank == 0:

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

    alt_beta_hats = np.zeros((num_tasks, n_features))
    alt_FNR_results = np.zeros(num_tasks)
    alt_FPR_results = np.zeros(num_tasks)
    alt_sa_results = np.zeros(num_tasks)
    alt_ee_results = np.zeros(num_tasks)
    alt_median_ee_results = np.zeros(num_tasks)

    record_alt = False

    # Risk calculations
    exact_risk = []
    MIC_risk = []

for i in range(num_tasks):
    start = time.time()

    f.seek(index[chunk_param_list[chunk_idx][i]], 0)
    params = pickle.load(f)

    # Skip this guy because for the particular combination of parameters,
    # all betas end up being zero
    if 'skip' in list(params.keys()):
        if params['skip']:
            continue

    params['comm'] = subcomm
    sigma = params['sigma']
    beta = params['betas']
    seed = params['seed']
    if subrank == 0:
        # Generate data
        X, X_test, y, y_test, ss = gen_data(params['n_samples'], params['n_features'],
                                        params['kappa'], sigma, beta, seed)
    else:
        X = None
        X_test = None
        y = None
        y_test = None
        ss = None


    X = Bcast_from_root(X, subcomm)
    X_test = Bcast_from_root(X_test, subcomm)
    y = Bcast_from_root(y, subcomm)
    y_test = Bcast_from_root(y_test, subcomm)
    ss = Bcast_from_root(ss, subcomm)

    params['ss'] = ss

    exp = locate('exp_types.%s' % exp_type)
    print('Going into exp')
    model = exp.run(X, y, params)

    if subrank == 0:
        #### Calculate and log results
        beta_hat = model.coef_.ravel()
        beta_hats[i, :] = beta_hat.ravel()
        fn_results[i] = np.count_nonzero(beta[beta_hat == 0, 0])
        fp_results[i] = np.count_nonzero(beta_hat[beta.ravel() == 0])
        r2_results[i] = r2_score(y_test, X_test @ beta)
        r2_true_results[i] = r2_score(y_test, model.predict(X_test))
    # Score functions have been modified, requiring us to first calculate log-likelihood

        llhood = log_likelihood_glm('normal', y_test, model.predict(X_test))
        try:
            BIC_results[i] = BIC(llhood, np.count_nonzero(beta_hat), y_test.size)
        except:
            BIC_results[i] = np.nan
        try:
            AIC_results[i] = AIC(llhood, np.count_nonzero(beta_hat))
        except:
            AIC_results[i] = np.nan
        try:
            AICc_results[i] = AICc(llhood, np.count_nonzero(beta_hat), y_test.size)
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

        if hasattr(model, 'alt_coef_'):
            record_alt = True
            beta_hat = model.alt_coef_.ravel()
            alt_beta_hats[i, :] = beta_hat.ravel()
            alt_FNR_results[i] = FNR(beta.ravel(), beta_hat)
            alt_FPR_results[i] = FPR(beta.ravel(), beta_hat)
            alt_sa_results[i] = selection_accuracy(beta.ravel(), beta_hat)
            ee, median_ee = estimation_error(beta.ravel(), beta_hat)
            alt_ee_results[i] = ee
            median_ee_results[i] = median_ee


        # Record the sresults on both the train and the test data
        MIC_risk_ = model.scores_.ravel()
        exact_risk_ = model.alt_scores_.ravel()

        exact_risk.append(exact_risk_)
        MIC_risk.append(MIC_risk_)

        print('Process group %d completed outer loop %d/%d' % (rank, i, num_tasks))
        print(time.time() - start)

    del params
    if args.test:
        break
# Gather across root nodes
if subrank == 0:

    v_list = [fn_results, fp_results, r2_results, r2_true_results, beta_hats, BIC_results,
             AIC_results, AICc_results, FNR_results, FPR_results, sa_results, ee_results,
             median_ee_results]

    for v in v_list:
        v = np.array(v)
        v = Gatherv_rows(v, roots_comm, root = 0)

    if record_alt:

        alt_v_list = [alt_beta_hats, alt_FNR_results, alt_FPR_results, alt_sa_results,
                      alt_ee_results, alt_median_ee_results]

        for v in alt_v_list:
            v = np.array(v)
            v = Gatherv_rows(v, roots_comm, root = 0)

    # Gather risk calculations
    pdb.set_trace()
    exact_risk = Gather_ndlist(exact_risk, roots_comm, root = 0)
    MIC_risk = Gather_ndlist(MIC_risk, roots_comm, root = 0)



f.close()
if comm.rank == 0:
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

        if record_alt:
            results['alt_beta_hats'] = alt_beta_hats
            results['alt_FNR_results'] = alt_FNR_results
            results['alt_FPR_results'] = alt_FPR_results
            results['alt_sa_results'] = alt_sa_results
            results['alt_ee_results'] = alt_ee_results
            results['alt_median_ee_results'] = alt_median_ee_results

        # Need to split up the list of arrays into separate datasets
        er = results.create_group('exact_risk')
        for i, exact_risk_ in enumerate(exact_risk):
            er.create_dataset(str(i), data = exact_risk_)
        mic = results.create_group('MIC_risk')
        for i, MIC_risk_ in enumerate(MIC):
            mic.create_dataset(str(i), data = MIC_risk_)


    print('Total time: %f' % (time.time() - total_start))
    print('Job completed!')

