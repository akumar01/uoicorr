import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
import importlib
from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from utils import gen_beta, gen_data, block_covariance

total_start = time.time()

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('--arg_file', default=None)
parser.add_argument('-l', action= 'store_true')
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


from pyuoi.linear_model.lasso import UoI_Lasso

### Import arguments from file ###
try:
	arg_file_path, arg_file = os.path.split(args.arg_file)
	sys.path.append(arg_file_path)
	args = importlib.import_module(arg_file) 
except:
	print('Warning! Could not load arg file')

n_features = args.n_features
block_size = args.block_size
kappa = args.kappa
est_score = args.est_score
reps = args.reps
correlations = args.correlations
selection_thres_mins = args.selection_thres_mins
sparsity = args.sparsity
results_file = args.results_file

# Ensure that selection_thres_mins and correlations are numpy arrays
if not isinstance(selection_thres_mins, np.ndarray):
	if np.isscalar(selection_thres_mins):
		selection_thres_mins = np.array([selection_thres_mins])
	else:
		selection_thres_mins = np.array(selection_thres_mins)

if not isinstance(correlations, np.ndarray):
	if np.isscalar(correlations):
		correlations = np.array([correlations])
	else:
		correlations = np.array(correlations)

results = h5py.File(results_file, 'w')

# result arrays: 
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, correlations.size, selection_thres_mins.size, n_features))

# result arrays: scores
fn_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
fp_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_true_results = np.zeros((reps, correlations.size))

for rep in range(reps):

	# Generate model coefficients (CHANGE SO BETADIST IS A FILE PARAM)
	beta = gen_beta(n_features, block_size, sparsity, betadist = 'uniform')

	for corr_idx, correlation in enumerate(correlations):

		# Block covariance structure
		Sigma = block_covariance(n_features, correlation, block_size)

		# Generate data (CHANGE SO N_SAMPLES IS A FILE PARAM)
		X, X_test, y, y_test = gen_data(n_samples, n_features, kappa,
												Sigma, beta)

		for thres_idx, selection_thres_min in enumerate(selection_thres_mins):
			start = time.time()
			uoi = UoI_Lasso(
				normalize=True,
				n_boots_sel=48,
				n_boots_est=48,
				estimation_score=args.est_score)
			uoi.fit(X, y.ravel())
			beta_hat = uoi.coef_
			beta_hats[rep, corr_idx, thres_idx, :] = beta_hat
			fn_results[rep, corr_idx, thres_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
			fp_results[rep, corr_idx, thres_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
			r2_results[rep, corr_idx, thres_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
			print(time.time() - start)
		r2_true_results[rep, corr_idx] = r2_score(y_test, np.dot(X_test, beta))

results['fn'] = fn_results
results['fp'] = fp_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results['beta'] = betas
results['beta_hats'] = beta_hats

results.close()
print('Total runtime: %f' %  (time.time() - total_start))

