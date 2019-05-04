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
from sklearn.covariance import oas
from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm
from pyuoi.mpi_utils import Bcast_from_root, Gatherv_rows
from utils import gen_data
from utils import FNR, FPR, selection_accuracy, estimation_error

p = parser = argparse.ArgumentParser()

parser.add_argument('data_file')
parser.add_argument('results_file', default = 'results.h5')
args = parser.parse_args()

results_file = args.results_file

# Create an MPI comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()    

for exp_type in ['CV_Lasso', 'EN', 'UoILasso', 'UoIElasticNet']:

    f = open(args.data_file, 'rb')
    n_reps = pickle.load(f)
    stratified = pickle.load(f)
    model_coefs = []
    for rep in range(n_reps):
        if stratified:
            groups = pickle.load(f)
        else:
            groups = None
        data = pickle.load(f)

        # Estimate covariance
        sigma_hat = oas(data)
        
        # Split data into training and test samples

        train_data, test_data = train_test_split(data, train_size = 0.8, stratify=groups,
                                                random_state = 0)

        model_coefs.append(np.zeros((data.shape[1], data.shape[1] - 1)))
        # Iterate over the all neurons
        for i in range(data.shape[1]):

            # Load params with the needed information
            p = {}
            p['comm'] = comm
            p['n_alphas'] = 48
            p['n_boots_sel'] = 48
            p['n_boots_est'] = 48
            p['est_score'] = 'r2'
            p['stability_selection'] = 1
            p['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
            exp = locate('exp_types.%s' % exp_type)
            X = train_data[:, np.arange(data.shape[1]) != i]
            model = exp.run(X, train_data[:, i], p, groups)
            
            model_coefs[i, :] = model.coef_
        # Fix exp_types to accept a group kfold if provided with groups
    model_coefs = np.array(model_coefs)

    # Save data
    with h5py.File('%s_%s.h5' % (exp_type, results_file), 'w') as f:
        f['coefs'] = model_coefs