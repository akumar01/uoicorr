import warnings
import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
import importlib
from scipy.linalg import block_diag
from sklearn.metrics import r2_score
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

from utils import gen_data, gen_beta, block_covariance

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

n_features = args.n_features
block_size = args.block_size
kappa = args.kappa
est_score = args.est_score
reps = args.reps
correlations = args.correlations
sparsity = args.sparsity
results_file = args.results_file
n_samples = args.n_samples
betadist = args.betadist

# Elastic Net hyperparmeters
l1_ratios = args.l1_ratios
n_alphas = args.n_alphas
cv_splits = 10

n_blocks = int(n_features/block_size)

# Ensure that l1_ratios and correlations are numpy arrays
if not isinstance(l1_ratios, np.ndarray):
	if np.isscalar(l1_ratios):
		l1_ratios = np.array([l1_ratios])
	else:
		l1_ratios = np.array(l1_ratios)

if not isinstance(correlations, np.ndarray):
	if np.isscalar(correlations):
		correlations = np.array([correlations])
	else:
		correlations = np.array(correlations)


results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, correlations.size, n_features))
beta_hats_cv = np.zeros((l1_ratios.size, n_alphas, n_features))
# result arrays: scores
fn_results = np.zeros((reps, correlations.size))
fp_results = np.zeros((reps, correlations.size))
r2_results = np.zeros((reps, correlations.size))
r2_true_results = np.zeros((reps, correlations.size))

for rep in range(reps):

	beta = gen_beta(n_features, block_size, sparsity, betadist=betadist)
	betas[rep, :] = beta.ravel()

	for corr_idx, correlation in enumerate(correlations):

		Sigma = block_covariance(n_features, correlation, block_size)

		X, X_test, y, y_test = gen_data(n_samples = n_samples, 
		n_features=n_features,	kappa = kappa, covariance = Sigma, beta = beta)

		alphas = np.zeros((l1_ratios.size, n_alphas))
		scores = np.zeros((l1_ratios.size, n_alphas))

		en = ElasticNet(normalize=True, warm_start = False)
		start = time.time()

		# Use 10 fold cross validation. Do this in a manual way to enable use of warm_start and custom parameter sweeps
		kfold = KFold(n_splits = cv_splits, shuffle = True)

		for l1_idx, l1_ratio in enumerate(l1_ratios):
			# Generate alphas to use
			alphas[l1_idx, :] = _alpha_grid(X = X, y = y.ravel(), l1_ratio = l1_ratio, normalize = True, n_alphas = n_alphas)

			for a_idx, alpha in enumerate(alphas[l1_idx, :]):

				en.set_params(alpha = alpha, l1_ratio = l1_ratio)

				cv_scores = np.zeros(cv_splits)
				# Cross validation splits into training and test sets
				for i, cv_idxs in enumerate(kfold.split(X, y)):
					en.fit(X[cv_idxs[0], :], y[cv_idxs[0]])
					cv_scores[i] = r2_score(y[cv_idxs[1]], en.coef_ @ X[cv_idxs[1], :].T)

				# Average together cross-validation scores
				scores[l1_idx, a_idx] = np.mean(cv_scores)

		print(time.time() - start)
		# Select the model with the maximum score
		max_score_idx = np.argmax(scores.ravel())
		max_score_idxs = np.unravel_index(max_score_idx, (l1_ratios.size, n_alphas))
		en.set_params(l1_ratio = l1_ratios[max_score_idxs[0]], alpha = alphas[max_score_idxs[0], max_score_idxs[1]])
		en.fit(X, y.ravel())
		beta_hat = en.coef_

		beta_hats[rep, corr_idx, :] = beta_hat
		fn_results[rep, corr_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
		fp_results[rep, corr_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
		r2_results[rep, corr_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
		r2_true_results[rep, corr_idx] = r2_score(y_test, np.dot(X_test, beta))

results.close()
print('Total runtime: %f' %  (time.time() - total_start))

