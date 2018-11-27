import argparse
import numpy as np
import h5py
import time

from scipy.linalg import block_diag

from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNetCV

from PyUoI.UoI_Lasso import UoI_Lasso

### parse arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--block_size', type=int, default=5)
parser.add_argument('--n_blocks', type=int, default=5)
parser.add_argument('--kappa', type=float, default=0.3)
parser.add_argument('--LST', type=float, default=0.75)
parser.add_argument('--reps', type=int, default=50)
parser.add_argument('--sparsity', type=float, default=1.)
parser.add_argument('--results_file', default='results.h5')
args = parser.parse_args()

# size of each block
block_size = args.block_size
# number of blocks
n_blocks = args.n_blocks
# inverse signal-to-noise ratio
kappa = args.kappa
# sparsity of within block features
sparsity = args.sparsity
# number of repetitions
reps = args.reps
# lower selection threshold
LST = args.LST
# filename for results
results_file = args.results_file

# set up other variables
n_features = block_size * n_blocks
n_samples = 5 * n_features
n_nonzero_beta = int(sparsity * block_size)

# correlations and selection thresholds
correlations = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# set up results file
results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats_uoi = np.zeros((reps, correlations.size, n_features))
beta_hats_enet = np.zeros((reps, correlations.size, n_features))
# result arrays: explained variance performance
r2_uoi = np.zeros((reps, correlations.size))
r2_enet = np.zeros((reps, correlations.size))
r2_true = np.zeros((reps, correlations.size))

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
		# apply uoi lasso
		start = time.time()
		uoi = UoI_Lasso(
			normalize=True,
			n_boots_sel=48,
			n_boots_est=48,
			selection_thres_min=LST,
			n_selection_thres=48,
			estimation_score='BIC'
		)
		uoi.fit(X, y.ravel())
		beta_hat_uoi = uoi.coef_
		print('uoi: ', time.time() - start)
		start = time.time()
		# apply elastic net
		enet = ElasticNetCV(
			l1_ratio=[0.01, .1, .5, .7, .9, .95, .99, 1],
			normalize=True, 
			tol=1e-7, 
			max_iter = 100000
		)
		enet.fit(X, y.ravel())
		beta_hat_enet = enet.coef_
		print('enet: ', time.time() - start)
		# store fits
		beta_hats_uoi[rep, corr_idx, :] = beta_hat_uoi
		beta_hats_enet[rep, corr_idx, :] = beta_hat_enet
		# calculate test performance
		r2_uoi[rep, corr_idx] = r2_score(
			y_test, np.dot(X_test, beta_hat_uoi)
		)
		r2_enet[rep, corr_idx] = r2_score(
			y_test, np.dot(X_test, beta_hat_enet)
		)
		r2_true[rep, corr_idx] = r2_score(
			y_test, np.dot(X_test, beta)
		)
# store results in h5 file
results['betas'] = betas
results['beta_hats_uoi'] = beta_hats_uoi
results['beta_hats_enet'] = beta_hats_enet
results['r2_uoi'] = r2_uoi
results['r2_enet'] = r2_enet
results['r2_true'] = r2_true
results.close()


