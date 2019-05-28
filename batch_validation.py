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
parser.add_argument('results_file')
parser.add_argument('exp_type')
parser.add_argument('-i', '--idxs', nargs = '+', type= int)
args = parser.parse_args()

data_file = args.data_file
results_file = args.results_file
exp_type = args.exp_type
idxs = args.idxs

f = open(data_file, 'rb')
train_data = pickle.load(f)
test_data = pickle.load(f)


model_coefs = np.zeros((len(idxs), train_data.shape[1] - 1))
selection_ratios = np.zeros(len(idxs))
r2 = np.zeros(len(idxs))
BIC_results = np.zeros(len(idxs))

# Covariance Estimation
sigma_train = oas(train_data)[0]
sigma_test = oas(test_data)[0]
sigma_full = oas(np.vstack((train_data, test_data)))[0]

p = {}
p['n_alphas'] = 48
p['n_boots_sel'] = 48
p['n_boots_est'] = 48
p['est_score'] = 'r2'
p['stability_selection'] = 1.0
p['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
p['reg_params'] = {}
p['reg_params']['lambda_S'] = np.linspace(0, 1, 10)
p['reg_params']['lambda_1'] = np.linspace(0, 1, 10)
p['reg_params']['lambda_TV'] = np.linspace(0, 1, 10)
p['sigma'] = sigma_train

for i, idx in enumerate(idxs):
    t0 = time.time()
    exp = locate('exp_types.%s' % exp_type)
    X = train_data[:, np.arange(train_data.shape[1]) != idx]
    X_test = test_data[:, np.arange(test_data.shape[1]) != idx]
    y = train_data[:, idx]
    y_test = test_data[:, idx]
    model = exp.run(X, y, p)
    model_coefs[i, :] = model.coef_
    selection_ratios[i] = np.count_nonzero(model.coef_)/model.coef_.size
    r2[i] = model.score(X_test, y_test)
    llhood = log_likelihood_glm('normal', y_test, model.predict(X_test))
    BIC_results[i] = BIC(llhood, np.count_nonzero(model.coef_), X_test.shape[0])
    print('iteration %d, time: %f' % (i, time.time() - t0))
# Save data
with h5py.File(results_file, 'w') as f:
    f['coefs'] = model_coefs
    f['sigma_train'] = sigma_train
    f['sigma_test'] = sigma_test
    f['sigma_full'] = sigma_full
    f['selection_ratios'] = selection_ratios
    f['r2'] = r2
    f['BIC'] = BIC_results
