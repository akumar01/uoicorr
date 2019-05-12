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
from sklearn.model_selection import train_test_split, KFold
from sklearn.covariance import oas
from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm
from pyuoi.mpi_utils import Bcast_from_root, Gatherv_rows
from utils import gen_data
from utils import FNR, FPR, selection_accuracy, estimation_error

parser = argparse.ArgumentParser()

parser.add_argument('data_file')
parser.add_argument('results_file', default = 'results.h5')
args = parser.parse_args()

results_file = args.results_file

# Create an MPI comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()    

outer_folds = 5

for exp_type in ['CV_Lasso', 'EN', 'UoILasso', 'UoIElasticNet']:
    print(exp_type)
    f = open(args.data_file, 'rb')
    n_reps = pickle.load(f)
    stratified = pickle.load(f)
    model_coefs = []
    sigma_hats = []
    selection_ratios = []
    r2 = []
    BIC_results = []
    for rep in range(n_reps):
        if stratified:
            groups = pickle.load(f)
        data = pickle.load(f)
        if not stratified:
            groups = np.zeros(data.shape[0])

        folds = KFold(n_splits = outer_folds, shuffle=True, random_state=0)

        coefs = np.zeros((outer_folds, data.shape[1], data.shape[1] - 1))
        selection_ratios_ = np.zeros((outer_folds, data.shape[1]))
        r2_ = np.zeros((outer_folds, data.shape[1]))
        BIC_ = np.zeros((outer_folds, data.shape[1]))        
        cv_idx = 0
        for train_index, test_index in folds.split(data):
            print('Fold %d' % cv_idx)
            train_data = data[train_index, :]
            test_data = data[test_index, :]
            # Iterate over the all neurons
            for i in range(data.shape[1]):
                print('Working on %d/%d' % (i, data.shape[1] - 1
                    ))
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
                X_test = test_data[:, np.arange(data.shape[1]) != 0]
                model = exp.run(X, train_data[:, i], p)
                # Record selection ratios, r^2, and BIC
                if rank == 0:
                    coefs[cv_idx, i, :] = model.coef_
                    selection_ratios_[cv_idx, i] = np.count_nonzero(model.coef_)/model.coef_.size
                    r2_[cv_idx, i] = r2_score(test_data[:, i], model.predict(X_test))
                    llhood = log_likelihood_glm('normal', test_data[:, i], model.predict(X_test))
                    BIC_[cv_idx, i] = BIC(llhood, np.count_nonzero(model.coef_), X_test.shape[0])
                    
            cv_idx += 1
        # Fix exp_types to accept a group kfold if provided with groups
        if rank == 0:
            model_coefs.append(coefs)
            # Estimate covariance
            cov, _ = oas(data)
            sigma_hats.append(cov)
            selection_ratios.append(selection_ratios_)
            r2.append(r2_)
            BIC_results.append(BIC_)

    if rank == 0:
        model_coefs = np.array(model_coefs)
        sigma_hats = np.array(sigma_hats)

        # Save data
        with h5py.File('%s_%s.h5' % (exp_type, results_file), 'w') as f:
            f['coefs'] = model_coefs
            f['sigma_hats'] =  sigma_hats
            f['seletion_ratios'] = selection_ratios
            f['r2'] = r2
            f['BIC'] = BIC_results