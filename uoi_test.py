import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import argparse
import numpy as np
import h5py
import time
import pdb
from scipy.linalg import block_diag
from sklearn.metrics import r2_score


# Hack to be able to import PyUoI
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

from pyuoi.UoI_Lasso import UoI_Lasso

# Can we get PyUoI_Lasso to recover a simple model?
reps = 25
# Inverse signal to noise ratio
kappa = 0.1

sparsity = 0.2

selection_thres_mins = np.array([0.5, 1.0])

results_file = 'C:\\Users\\akumar\\nse\\uoicorr\\data\\test\\test6.h5'
assert not os.path.isfile(results_file), "Results file already exists!"

n_features = 20

n_nonzero_beta = int(sparsity * n_features)

n_samples = 5 * n_features

results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, selection_thres_mins.size, n_features))
# result arrays: scores
fn_results = np.zeros((reps, selection_thres_mins.size))
fp_results = np.zeros((reps, selection_thres_mins.size))
r2_results = np.zeros((reps, selection_thres_mins.size))
r2_true_results = np.zeros(reps)

for rep in range(reps):
	beta = np.random.uniform(low = 0, high = 10, size = (n_features, 1))
	mask = np.zeros(n_features)
	mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
	np.random.shuffle(mask)
	mask = mask[..., np.newaxis]
	beta = beta * mask
	betas[rep, :] = beta.ravel()
	Sigma = np.diag(np.ones(n_features))
	X = np.random.multivariate_normal(mean = np.zeros(n_features), cov = Sigma, size = n_samples)
	X_test = np.random.multivariate_normal(mean = np.zeros(n_features), cov = Sigma, size = n_samples)

	# signal and noise variance
	# signal and noise variance
	signal_variance = np.sum(Sigma * np.dot(beta, beta.T))
	noise_variance = kappa * signal_variance
	# draw noise
	noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
	noise_test = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
	# response variable
	y = np.dot(X, beta) + noise
	y_test = np.dot(X_test, beta) + noise_test
	start = time.time()
	for i, s in enumerate(selection_thres_mins):
		uoi = UoI_Lasso(
			normalize=True,
			n_boots_sel=48,
			n_boots_est=48,
			estimation_score='BIC',
			stability_selection = s
		)
		uoi.fit(X, y.ravel())
		beta_hat = uoi.coef_
		beta_hats[rep, i, :] = beta_hat
		fn_results[rep, i] = np.count_nonzero(beta[beta_hat == 0, 0])
		fp_results[rep, i] = np.count_nonzero(beta_hat[beta.ravel() == 0])
		r2_results[rep, i] = r2_score(y_test, np.dot(X_test, beta_hat))
	print(time.time() - start)
	r2_true_results[rep] = r2_score(y_test, np.dot(X_test, beta))
results['fn'] = fn_results
results['fp'] = fp_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results['beta'] = betas
results['beta_hats'] = beta_hats
results.close()
