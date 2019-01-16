import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
import json
from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from utils import gen_beta, gen_data

total_start = time.time()

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('arg_file', default=None)
parser.add_argument('-l', '--local', action= 'store_true')
args = parser.parse_args()

### Flag whether or not the script is being run locally ###
if args.l:
	# Hack to import pyuoi
	parent_path, current_dir = os.path.split(os.path.abspath('.'))
	while current_dir not in ['nse']:
		parent_path, current_dir = os.path.split(parent_path)
	p = os.path.join(parent_path, current_dir)
	# Add analysis
	if p not in sys.path:
		sys.path.append(p)

	# And standard list of subdirectories
	if '%s\\pyuoi' % p not in sys.path:
		sys.path.append('%s\\pyuoi' % p)

from pyuoi.linear_models.lasso import UoI_Lasso
from pyuoi.linear_models.lasso import UoI_ElasticNet

### Import arguments from file ###

arg_file_path, arg_file = os.path.split(args.arg_file)
sys.path.append(arg_file_path)
with open(arg_file, 'r') as f: 
	args = json.load(f)
	f.close()

# Unpack args
n_features = args['n_features']
block_size = args['block_size']
kappa = args['kappa']
est_score = args['est_score']
reps = args['reps']
correlations = args['correlations']
selection_thres_mins = args['selection_thres_mins']
sparsity = args['sparsity']
results_file = args['results_file']
betdist = args['betadist']
n_samples = args['n_samples']

exp_type = args['exp_type']

# 
from exp_classes import exp_type

# betas will be be changed only every repetition
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, correlations.size, n_features))

# result arrays: scores
fn_results = np.zeros((reps, correlations.size))
fp_results = np.zeros((reps, correlations.size))
r2_results = np.zeros((reps, correlations.size))
r2_true_results = np.zeros((reps, correlations.size))

for rep in range(reps):

	# Generate model coefficients
	beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)

	for corr_idx, correlation in enumerate(correlations):

		# Return covariance matrix
		sigma = exp_type.covariance 