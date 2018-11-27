import argparse
import numpy as np
import h5py
import time

from scipy.linalg import block_diag
from sklearn.metrics import r2_score

from PyUoI.UoI_Lasso import UoI_Lasso

### parse arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, default=20)
parser.add_argument('--kappa', type=float, default=0.3)
parser.add_argument('--reps', type=int, default=50)
parser.add_argument('--sparsity', type=float, default=1.)
parser.add_argument('--results_file', default='results.h5')
args = parser.parse_args()

# number of features
n_features = args.n_features
# inverse signal-to-noise ratio
kappa = args.kappa
# sparsity of within block features
sparsity = args.sparsity
# number of repetitions
reps = args.reps
# filename for results
results_file = args.results_file

# set up other variables
n_samples = 5 * n_features
n_nonzero_beta = int(sparsity * n_features)

# correlations and selection thresholds
Ls = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
selection_thres_mins = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats = np.zeros((reps, Ls.size, selection_thres_mins.size, n_features))
# result arrays: scores
fn_results = np.zeros((reps, Ls.size, selection_thres_mins.size))
fp_results = np.zeros((reps, Ls.size, selection_thres_mins.size))
r2_results = np.zeros((reps, Ls.size, selection_thres_mins.size))
r2_true_results = np.zeros((reps, Ls.size))

for rep in range(reps):
	beta = np.random.uniform(low=0, high=10, size=(n_features, 1))
	mask = np.zeros(n_features)
	mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
	np.random.shuffle(mask)
	beta = beta * mask[..., np.newaxis]
	betas[rep, :] = beta.ravel()
	for L_idx, L in enumerate(Ls):
		# create covariance matrix for block
		indices = np.arange(n_features)
		distances = np.abs(np.subtract.outer(indices, indices))
		Sigma = np.exp(-distances/L)
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
				selection_thres_min=selection_thres_min,
				n_selection_thres=48,
				estimation_score='BIC'
			)
			uoi.fit(X, y.ravel())
			beta_hat = uoi.coef_
			beta_hats[rep, L_idx, thres_idx, :] = beta_hat
			fn_results[rep, L_idx, thres_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
			fp_results[rep, L_idx, thres_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
			r2_results[rep, L_idx, thres_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
			print(time.time() - start)
		r2_true_results[rep, L_idx] = r2_score(y_test, np.dot(X_test, beta))
results['fn'] = fn_results
results['fp'] = fp_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results['beta'] = betas
results['beta_hats'] = beta_hats
results.close()


