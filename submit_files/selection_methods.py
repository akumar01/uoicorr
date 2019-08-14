import numpy as np
import itertools
import pdb
from misc import get_cov_list
from utils import gen_beta2

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

###### Master list of parameters to be iterated over #######

exp_types =  ['CV_Lasso', 'mcp']

# Estimated worst case run-time for a single repitition for each algorithm in exp_types 
algorithm_times = ['08:00:00', '01:00:00', '01:00:00', '01:00:00', '2:00:00']

n_features = 100

# Block sizes
block_sizes = [10, 20]

# Block correlation
correlation = [0, 0.08891397, 0.15811388, 0.28117066, 0.5]

# Exponential length scales
L = [1, 2, 5, 10]

cov_list, _ = get_cov_list(n_features, 14, correlation, block_sizes, L, n_supplement = 0)

cov_params = [{'correlation' : t[0], 'block_size' : t[1], 'L' : t[2], 't': t[3]} for t in cov_list]


iter_params = {

'cov_params' : cov_params,

}

#############################################################

##### DO NOT CHANGE THIS UNLESS YOU ARE SURE!!!! ############
betaseed = 1234

# For each desired betawidth, we generate a fixed beta vector
# that is held fixed across all paramter combinations. For a fixed
# sparsity, the exact imposed sparsity profile may vary depending 
# on the block size of the covariance matrix. However, we use the
# blocks as a seed for the shuffling that is done, so that for a fixed
# sparsity and block size, all beta vectors should be identical

betawidth = [0.1, np.inf, -1]

beta_dict = []
for i, bw in enumerate(betawidth):

	beta_dict.append({'betawidth' : bw, 'beta': gen_beta2(n_features, n_features, 1, bw, seed = betaseed)})

##### Common parameters held fixed across all jobs ##########
comm_params = {
'sparsity': sparsity,
'cov_type' : 'interpolation',
'n_features' : n_features,
# n/p ratio #
'np_ratio': 5,
'est_score': 'BIC',
'reps' : 5,
'stability_selection' : [1.0],
'n_boots_sel': 25,
'n_boots_est' : 25,
'betadict' : beta_dict,
# Inverse Signal to noise ratio
'kappa' : [5, 2, 1],
'sub_iter_params': ['betadict', 'sparsity', 'kappa']
}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1,  0.5, 0.75, 0.9, 0.95]
comm_params['n_alphas'] = 100

# Parameters for SCAD/MCP
comm_params['gamma'] = [3]

# Parameters for SLOPE
comm_params['lambda_method'] = 'FDR'
comm_params['lambda_args'] = np.linspace(0.01, 0.9, 50)

###############################################################

# Which selection methods should we apply to the algorithms?
comm_params['selection_methods'] = ['BIC', 'AIC', 'aBIC', 'eBIC', 'CV', 'mBIC', 'gMDL', 'empirical_bayes']
# Which fields should we record for each selection method? 
comm_params['fields'] = {'BIC' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param'], 
						 'aBIC' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param',
						 		   'sparsity_estimates', 'oracle_sa', 'oracle_FNR', 'oracle_FPR',
						 		   'oracle_ee', 'oracle_r2', 'oracle_MSE', 'oracle_penalty', 'effective_penalty'], 
						 'AIC' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param'], 
						 'eBIC' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param'], 
						 'CV' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param'],
                         'mBIC' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param'],
                         'gMDL' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 'effective_penalty'],
                         'empirical_bayes' : ['beta_hats', 'FNR', 'FPR', 'sa', 'ee', 'r2', 'MSE', 'reg_param', 
                                              'effective_penalty']}
