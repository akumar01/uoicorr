import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import argparse
import numpy as np
import h5py
import time
import pdb
from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from PyUoI.UoI_Lasso import UoI_Lasso

# Can we get PyUoI_Lasso to recover a simple model?

reps = 50
# Inverse signal to noise ratio
kappa = 0.1

sparsity = 0.2

results_file = 'C:\\Users\\akumar\\nse\\uoicorr\\data\\test\\test1.h5'

n_features = 20

n_nonzero_beta = int(sparsity * n_features)

n_samples = 5 * n_features

results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, correlations.size, selection_thres_mins.size, n_features))
# result arrays: scores
fn_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
fp_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_results = np.zeros((reps, correlations.size, selection_thres_mins.size))
r2_true_results = np.zeros((reps, correlations.size))

for rep in range(reps):
	beta = np.random.uniform(low = 10, high = 10, size = (n_features, 1))
	mask = np.array([])
	for block in range(n_blocks):
		block_mask = np.zeros(block_size)
		block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
		np.random.shuffle(block_mask)
		mask = np.concatenate((mask, block_mask))
	mask = mask[..., np.newaxis]
	beta = beta * mask
	betas[rep, :] = beta.ravel()
	X = np.random.multivariate_normal()
