import numpy as np
import itertools
import pdb
from misc import get_cov_list

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

desc = 'Consider increasingly larger sample sizes, calculating the exact risk, MIC on train and test sets. Additionally sweep over the sparsity\
		and '

exp_types =  ['UoILasso']
# Estimated worst case run-time for a single repitition for each algorithm in exp_types 
algorithm_times = ['6:00:00']

n_features = 100

# Block sizes
block_sizes = [20]

# Block correlation
correlation = [0, 0.08891397, 0.15811388, 0.28117066, 0.5]

# Exponential length scales
L = [2, 5, 10, 20]

cov_list, _ = get_cov_list(n_features, 45, correlation, block_sizes, L, n_supplement = 0)

cov_params = [{'correlation' : t[0], 'block_size' : t[1], 'L' : t[2], 't': t[3]} for t in cov_list]

iter_params = {
'cov_params' : cov_params
}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'sparsity' : np.linspace(0.05, 1, 15),
'manual_penalty' : np.linspace(0, 5, 50),
'cov_type' : 'interpolation',
'reg_params': [],
'n_models': 1,
'n_features' : n_features,
# n/p ratio #
'np_ratio': 5,
'est_score': 'MIC',
'reps' : 10,
'stability_selection' : 1.,
'n_boots_sel': 48,
'n_boots_est' : 48,
'betawidth' : np.inf,
# Inverse Signal to noise ratio
'kappa' : 5,
'sub_iter_params': ['sparsity', 'manual_penalty']
}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################
