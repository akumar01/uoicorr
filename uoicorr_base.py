import sys, os
import argparse
import numpy as np
import h5py
import time
import pdb
import json

from scipy.linalg import block_diag
from sklearn.metrics import r2_score
try:
	from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm
except:
	sys.path.append('../PyUoI')
	from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm	
from pydoc import locate

from utils import gen_beta, gen_data, gen_covariance
from utils import FNR, FPR, selection_accuracy, estimation_error

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

with open(args.arg_file, 'r') as f: 
	args = json.load(f)
	f.close()

# Unpack args
n_features = args['n_features']
block_size = args['block_size']
kappa = args['kappa']
est_score = args['est_score']
reps = args['reps']
sparsity = args['sparsity']
results_file = args['results_file']
betadist = args['betadist']
n_samples = args['n_samples']

if 'const_beta' in list(args.keys()):
	const_beta = args['const_beta']
else:
	const_beta = False

# Specify type of covariance matrix and which
# fitting procedure to use
cov_type = args['cov_type']
cov_params = args['cov_params']
exp_type = args['exp_type']

# Wrap dictionary in a list
if type(cov_params) != list:
	cov_params = [cov_params]

# Determines the type of experiment to do 
# exp = importlib.import_module(exp_type, 'exp_types')
exp = locate('exp_types.%s' % exp_type)

results = h5py.File(results_file, 'w')

# Use the n_models flags to allow experiments to return
# multiple models over multiple parameters
shape = (reps, len(cov_params), args['n_models'])

betas = np.zeros((reps, n_features))
beta_hats = np.zeros(shape + (n_features,))

# result arrays: scores
fn_results = np.zeros(shape)
fp_results = np.zeros(shape)
r2_results = np.zeros(shape)
r2_true_results = np.zeros(shape)

BIC_results = np.zeros(shape)
AIC_results = np.zeros(shape)
AICc_results = np.zeros(shape)

FNR_results = np.zeros(shape)
FPR_results = np.zeros(shape)
sa_results = np.zeros(shape)
ee_results = np.zeros(shape)
median_ee_results = np.zeros(shape)

# Keep model coefficients fixed across repititions
if const_beta:
	beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)	

for rep in range(reps):

	# Generate new model coefficients for each repitition
	if not const_beta:
		beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)
	betas[rep, :] = beta.ravel()

	for cov_idx, cov_param in enumerate(cov_params):
		start = time.time()
		# Return covariance matrix
		# If the type of covariance is interpolate, then the matricies have been
		# pre-generated
		if cov_type == 'interpolate':
			sigma = np.array(cov_param['sigma'])
		else:
			sigma = gen_covariance(cov_type, n_features, block_size, **cov_param)
		X, X_test, y, y_test = gen_data(n_samples = n_samples, 
		n_features= n_features,	kappa = kappa, covariance = sigma, beta = beta)

		if exp_type == 'GTV':
			args['cov'] = sigma
		models = exp.run(X, y, args)

		for i, m in enumerate(models):
			beta_hat = m.coef_
			beta_hats[rep, cov_idx, i, :] = beta_hat.ravel()
			fn_results[rep, cov_idx, i] = np.count_nonzero(beta[beta_hat == 0, 0])
			fp_results[rep, cov_idx, i] = np.count_nonzero(beta_hat[beta.ravel() == 0])
			r2_results[rep, cov_idx, i] = r2_score(y_test, np.dot(X_test, beta_hat))
			r2_true_results[rep, cov_idx, i] = r2_score(y_test, np.dot(X_test, beta))

			# Score functions have been modified, requiring us to first calculate log-likelihood
			llhood = log_likelihood_glm('normal', y_test, np.dot(X_test, beta))

			BIC_results[rep, cov_idx, i] = BIC(llhood, np.count_nonzero(beta_hat), n_samples)
			AIC_results[rep, cov_idx, i] = AIC(llhood, np.count_nonzero(beta_hat))
			AICc_results[rep, cov_idx, i] = AICc(llhood, np.count_nonzero(beta_hat), n_samples)

			# Perform calculation of FNR, FPR, selection accuracy, and estimation error
			# here:

			FNR_results[rep, cov_idx, i] = FNR(beta.ravel(), beta_hat)
			FPR_results[rep, cov_idx, i] = FPR(beta.ravel(), beta_hat)
			sa_results[rep, cov_idx, i] = selection_accuracy(beta.ravel(), beta_hat)
			ee, median_ee = estimation_error(beta.ravel(), beta_hat)
			ee_results[rep, cov_idx, i] = ee
			median_ee_results[rep, cov_idx, i] = median_ee


		print(time.time() - start)

results['fn'] = fn_results
results['fp'] = fp_results
results['r2'] = r2_results
results['r2_true'] = r2_true_results
results['betas'] = betas
results['beta_hats'] = beta_hats
results['BIC'] = BIC_results
results['AIC'] = AIC_results
results['AICc'] = AICc_results

results['FNR'] = FNR_results
results['FPR'] = FPR_results
results['sa'] = sa_results
results['ee'] = ee_results
results['median_ee'] = median_ee_results

# import pickle
# pickle_file = results_file.split('.h5')[0]
# with open('%s2' % pickle_file, 'wb') as f:
# 	pickle.dump(selection_coefs, f)

results.close()
print('Total runtime: %f\n' %  (time.time() - total_start))
print('Job completed!')