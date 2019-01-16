import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
import json
import importlib

from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from pydoc import locate

from utils import gen_beta, gen_data

total_start = time.time()

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('arg_file', default=None)
parser.add_argument('-l', '--local', action= 'store_true')
args = parser.parse_args()

### Flag whether or not the script is being run locally ###
# Due to local directory structure, importing PyUoI requires adding
# some paths
if args.local:
	parent_path, current_dir = os.path.split(os.path.abspath('.'))
	while current_dir not in ['nse']:
		parent_path, current_dir = os.path.split(parent_path)
	p = os.path.join(parent_path, current_dir)

	if p not in sys.path:
		sys.path.append(p)

	if '%s\\pyuoi' % p not in sys.path:
		sys.path.append('%s\\pyuoi' % p)

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
betadist = args['betadist']
n_samples = args['n_samples']

exp_type = args['exp_type']
args['l1_ratios'] = [0.5]
# Determines the type of experiment to do 
# exp = importlib.import_module(exp_type, 'exp_types')
exp = locate('exp_types.%s' % exp_type)


results = h5py.File(results_file, 'w')

# betas will be be changed only every repetition
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, len(correlations), n_features))

# result arrays: scores
fn_results = np.zeros((reps, len(correlations)))
fp_results = np.zeros((reps, len(correlations)))
r2_results = np.zeros((reps, len(correlations)))
r2_true_results = np.zeros((reps, len(correlations)))

for rep in range(reps):

	# Generate model coefficients
	beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)

	for corr_idx, correlation in enumerate(correlations):
		start = time.time()
		# Return covariance matrix
		sigma = exp.get_covariance(n_features, correlation, block_size)

		X, X_test, y, y_test = gen_data(n_samples = n_samples, 
		n_features=n_features,	kappa = kappa, covariance = sigma, beta = beta)

		model = exp.run(X, y, args)

		beta_hat = model.coef_
		fn_results[rep, corr_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
		fp_results[rep, corr_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
		r2_results[rep, corr_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
		r2_true_results[rep, corr_idx] = r2_score(y_test, model.predict(X_test))

		print(time.time() - start)

results['fn'] = fn_results
results['fp'] = fp_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results['beta'] = betas
results['beta_hats'] = beta_hats

results.close()
print('Total runtime: %f' %  (time.time() - total_start))
