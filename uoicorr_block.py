import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
from scipy.linalg import block_diag
from sklearn.metrics import r2_score
import importlib
import pdb
from pyuoi.linear_model.lasso import UoI_Lasso

total_start = time.time()

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('--arg_file', default=None)

### Import arguments from file ###
try:
	arg_file_path, arg_file = os.path.split(parser.parse_args().arg_file)
	sys.path.append(arg_file_path)
	args = importlib.import_module(arg_file) 
except:
	print('Warning! Could not load arg file')
block_size = args.block_size
n_blocks = args.n_blocks
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

# set up other variables
n_features = block_size * n_blocks
n_samples = 5 * n_features
n_nonzero_beta = int(sparsity * block_size)

results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, correlations.size, selection_thres_mins.size, n_features))
# result arrays: scores
fn_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
fp_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_true_results = np.zeros((reps, correlations.size))

sigmas = np.zeros((reps, correlations.size, n_features, n_features))

for rep in range(reps):
	beta = np.random.uniform(low=0, high=10, size=(n_features, 1))
	mask = np.array([])
	for block in range(n_blocks):
		block_mask = np.zeros(block_size)
		block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
		np.random.shuffle(block_mask)
		mask = np.concatenate((mask, block_mask))
	mask = mask[..., np.newaxis]
	beta = beta * mask
	betas[rep, :] = beta.ravel()
	for corr_idx, correlation in enumerate(correlations):
		# create covariance matrix for block
		block_Sigma = correlation * np.ones((block_size, block_size)) 
		np.fill_diagonal(block_Sigma, np.ones(block_size))
		# populate entire covariance matrix
		rep_block_Sigma = [block_Sigma] * n_blocks
		Sigma = block_diag(*rep_block_Sigma)
		sigmas[rep, corr_idx] = Sigma
		# draw samples
		X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_samples)
		X_test = np.random.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_samples)

		# signal and noise variance
		signal_variance = np.sum(Sigma * np.dot(beta, beta.T))
		noise_variance = kappa * signal_variance
		# draw noise
		noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
		noise_test = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
		# response variable
		y = np.dot(X, beta) + noise
		y_test = np.dot(X_test, beta) + noise_test

		for thres_idx, selection_thres_min in enumerate(selection_thres_mins):
			start = time.time()
			uoi = UoI_Lasso(
				normalize=True,
				n_boots_sel=48,
				n_boots_est=48,
				estimation_score=args.est_score,
				stability_selection = selection_thres_min
			)
			uoi.fit(X, y.ravel())
			beta_hat = uoi.coef_
			beta_hats[rep, corr_idx, thres_idx, :] = beta_hat
			fn_results[rep, corr_idx, thres_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
			fp_results[rep, corr_idx, thres_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
			r2_results[rep, corr_idx, thres_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
			print(time.time() - start)
		r2_true_results[rep, corr_idx] = r2_score(y_test, np.dot(X_test, beta))

results['sigma'] = sigmas
results['fn'] = fn_results
results['fp'] = fp_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results['beta'] = betas
results['beta_hats'] = beta_hats
results.close()
print('Total runtime: %f' %  (time.time() - total_start))

