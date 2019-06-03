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
from pyuoi.mpi_utils import Bcast_from_root, Gatherv_rows
from utils import gen_data
from utils import FNR, FPR, selection_accuracy, estimation_error

total_start = time.time()

###### Command line arguments #######
parser = argparse.ArgumentParser()

parser.add_argument('arg_file')
parser.add_argument('results_file', default = 'results.h5')
parser.add_argument('exp_type', default = 'UoILasso')
parser.add_argument('--comm_splits', type=int, default = None)
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

# Only record the selection accuracy of estimates
if subrank == 0:
    
    estimates_sa = []

for i in range(num_tasks):
    start = time.time()
    
    f.seek(index[chunk_param_list[chunk_idx][i]], 0)
    params = pickle.load(f)
    params['comm'] = subcomm
    sigma = params['sigma']
    beta = params['betas']
    seed = params['seed']
    if subrank == 0:
        # Generate data
        X, X_test, y, y_test, _ = gen_data(params['n_samples'], params['n_features'],
                                        params['kappa'], sigma, beta, seed)
    else:
        X = None
        X_test = None
        y = None 
        y_test = None
        
    X = Bcast_from_root(X, subcomm)
    X_test = Bcast_from_root(X_test, subcomm)
    y = Bcast_from_root(y, subcomm)
    y_test = Bcast_from_root(y_test, subcomm)
    exp = locate('exp_types.%s' % exp_type)
    print('Going into exp')
    model = exp.run(X, y, params)

    if subrank == 0:
        #### Calculate and log results
        n_boots_sel, n_supports, n_coef = model.estimates_.shape

        sa = np.zeros((n_boots_sel, n_supports))

        for i1 in range(n_boots_sel):
            for i2 in range(n_supports):
                sa[i1, i2] = selection_accuracy(beta.ravel(), model.estimates_[i1, i2, :])

        estimates_sa.append(sa)

        print('Process group %d completed outer loop %d/%d' % (rank, i, num_tasks -1))
        print(time.time() - start)
    del params
    
# No Gather across root nodes, assuming that there is only one root node
f.close()    


if comm.rank == 0:
    # Pickle away results
    with open(results_file, 'wb') as results:
        results.write(pickle.dumps(estimates_sa))

    print('Total time: %f' % (time.time() - total_start))
    print('Job completed!')
    
