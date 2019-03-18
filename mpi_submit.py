import sys, os
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse
import json
import importlib
import subprocess
import numpy as np
from mpi4py import MPI

from scipy.linalg import block_diag
from sklearn.metrics import r2_score
from pyuoi.utils import BIC, AIC, AICc, log_likelihood_glm
from pyuioi.mpi_utils import Bcast_from_root
from pydoc import locate

from utils import gen_beta, gen_data, gen_covariance
from utils import FNR, FPR, selection_accuracy, estimation_error

if __name__ == '__main__':
	
	###### Command line arguments #######
	parser = argparse.ArgumentParser()
	
	# Param file from which to create job scripts
	parser.add_argument('arg_file', default=None)

	# Edit the 'reps' parameter to equal 1
	parser.add_argument('-s', '--single_rep', action='store_true')

	# Run the job on interactive nodes
	parser.add_argument('-i', '--interactive', action='store_true')
	#######################################

	# Load param file
	with open(args.arg_file, 'r') as f: 
		args = json.load(f)
		f.close()

	## Initialize a bunch of stuff ###
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
	shape = (reps, len(cov_params))

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

	# Create an MPI comm object
	comm = MPI.COMM_WORLD

	rank = comm.rank

	# This will be fed into call to UoI	
	args['comm'] = comm

	# Keep beta fixed across repetitions
	if const_beta:
		beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)

	for rep in range(reps):

		# Generate new model coefficients for each repetition
		if not const_beta:
			beta = gen_beta(n_features, block_size, sparsity, betadist = betadist)
		betas[rep, :] = beta.ravel()

		for cov_idx, cov_param in enumerate(cov_params):
			start = time.time()
			if rank == 0:
				# Return covariance matrix
				# If the type of covariance is interpolate, then the matricies have been
				# pre-generated
				if cov_type == 'interpolate':
					sigma = np.array(cov_param['sigma'])
				else:
					sigma = gen_covariance(cov_type, n_features, block_size, **cov_param)
				X, X_test, y, y_test = gen_data(n_samples = n_samples, 
				n_features= n_features,	kappa = kappa, covariance = sigma, beta = beta)

			else:
				X = None
				y = None

			X = Bcast_from_root(X, comm, root = 0)
			y = Bcast_from_root(y, comm, root = 0)

			# Call to UoI
			model = exp.run(X, y, args)
			if rank == 0:
				#### Calculate and log results
				beta_hat = model[0].coef_
				beta_hats[rep, cov_idx, :] = beta_hat.ravel()
				fn_results[rep, cov_idx] = np.count_nonzero(beta[beta_hat == 0, 0])
				fp_results[rep, cov_idx] = np.count_nonzero(beta_hat[beta.ravel() == 0])
				r2_results[rep, cov_idx] = r2_score(y_test, np.dot(X_test, beta_hat))
				r2_true_results[rep, cov_idx] = r2_score(y_test, np.dot(X_test, beta))
				# Score functions have been modified, requiring us to first calculate log-likelihood
				llhood = log_likelihood_glm('normal', y_test, np.dot(X_test, beta))
				try:
					BIC_results[rep, cov_idx] = BIC(llhood, np.count_nonzero(beta_hat), n_samples)
				except:
					BIC_results[rep, cov_idx] = np.nan
				try:
					AIC_results[rep, cov_idx] = AIC(llhood, np.count_nonzero(beta_hat))
				except:
					AIC_results[rep, cov_idx] = np.nan
				try:
					AICc_results[rep, cov_idx] = AICc(llhood, np.count_nonzero(beta_hat), n_samples)
				except:
					AICc_results[rep, cov_idx] = np.nan

				# Perform calculation of FNR, FPR, selection accuracy, and estimation error
				# here:
				FNR_results[rep, cov_idx] = FNR(beta, beta_hat)
				FPR_results[rep, cov_idx] = FPR(beta, beta_hat)
				sa_results[rep, cov_idx] = selection_accuracy(beta, beta_hat)
				ee, median_ee = estimation_error(beta, beta_hat)
				ee_results[rep, cov_idx] = ee
				median_ee_results[rep, cov_idx] = median_ee

	if rank == 0:
		# Save results
		results = h5py.File(results_file, 'w')
		results['beta_hat'] = beta_hat
		results.close()