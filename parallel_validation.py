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
parser.add_argument('results_file')
parser.add_argument('exp_type')
args = parser.parse_args()

results_file = args.results_file

# Create an MPI comm world object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()    

f = open(args.data_file, 'rb')
idxs = pickle.load(f)
train_data = pickle.load(f)
test_data = pickle.load(f)

model_coefs = np.zeros((len(idxs), train_data.shape[1] - 1))
selection_ratios = np.zeros(len(idxs))
r2 = np.zeros(len(idxs))
BIC_results = np.zeros(len(idxs))

p = {}
p['comm'] = comm
p['n_alphas'] = 48
p['n_boots_sel'] = 48
p['n_boots_est'] = 48
p['est_score'] = 'r2'
p['stability_selection'] = 1
p['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]

for idx in idxs:

	exp = locate('exp_types.%s' % exp_type)
	X = train_data[:, np.arange(data.shape[1]) != idx]
	X_test = test_data[:, np.arange(data.shape[1]) != 0]
	model = exp.run(X, train_data[:, idx], p)

	if rank == 0:
		model_coefs.append(coefs)
		

