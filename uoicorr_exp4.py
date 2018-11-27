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
parser.add_argument('--n_features', type=int, default=20)
parser.add_argument('--kappa', type=float, default=0.3)
parser.add_argument('--reps', type=int, default=50)
parser.add_argument('--sparsity', type=float, default=1.)
parser.add_argument('--LST', type=float, default=0.75)
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
# lower selection threshold
LST = args.LST
# filename for results
results_file = args.results_file

# set up other variables
n_samples = 5 * n_features # number of training samples
n_nonzero_beta = int(sparsity * n_features) # number of nonzero parameters

# correlations and selection thresholds
Ls = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# set up results file
results = h5py.File(results_file, 'w')
# result arrays: fits
betas = np.zeros((reps, n_features))
beta_hats_uoi = np.zeros((reps, Ls.size, n_features))
beta_hats_enet = np.zeros((reps, Ls.size, n_features))
# result arrays: explained variance performance
r2_uoi = np.zeros((reps, Ls.size))
r2_enet = np.zeros((reps, Ls.size))
r2_true = np.zeros((reps, Ls.size))

for rep in range(reps):
	# choose true parameters
	beta = np.random.uniform(low=0, high=10, size=(n_features, 1))
	# determine sparsity indicies
	mask = np.zeros(n_features)
	mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
	np.random.shuffle(mask)
	# apply mask to parameters to set them equal to zero
	beta = beta * mask[..., np.newaxis]
	betas[rep, :] = beta.ravel()
	# iterate over correlation strengths
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
		beta_hats_uoi[rep, L_idx, :] = beta_hat_uoi
		beta_hats_enet[rep, L_idx, :] = beta_hat_enet
		# calculate test performance
		r2_uoi[rep, L_idx] = r2_score(
			y_test, np.dot(X_test, beta_hat_uoi)
		)
		r2_enet[rep, L_idx] = r2_score(
			y_test, np.dot(X_test, beta_hat_enet)
		)
		r2_true[rep, L_idx] = r2_score(
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