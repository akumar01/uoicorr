import numpy as np
import itertools
import pdb
from misc import get_cov_list

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

desc = 'Benchmark with inverse exponentially distriubted coefficients to benchmark MIC\
		approaches against'

exp_types =  ['CV_Lasso, EN']
# Estimated worst case run-time for a single repitition for each algorithm in exp_types 
algorithm_times = ['02:00:00', '04:00:00']

n_features = 100

# Manual spread of cov_params
cov_params = [{'correlation': 0, 'block_size': 20, 'L': 1, 't': 0},
       {'correlation': 0.08891397, 'block_size': 20, 'L': 1, 't': 0},
       {'correlation': 0.28117066, 'block_size': 20, 'L': 1, 't': 0},
       {'correlation': 0.5, 'block_size': 20, 'L': 1, 't': 0},
       {'correlation': 1, 'block_size': 100, 'L': 2, 't': 1},
       {'correlation': 1, 'block_size': 100, 'L': 5, 't': 1},
       {'correlation': 1, 'block_size': 100, 'L': 10, 't': 1},
       {'correlation': 1, 'block_size': 100, 'L': 20, 't': 1},
       {'correlation': 0.5, 'block_size': 20, 'L': 20, 't': 0.9506632753385218}]

iter_params = {
'cov_params' : cov_params
}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'sparsity' : np.linspace(0.05, 1, 15),
'cov_params' : cov_params,
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
# Negative betawidth gives inverse exponential distribution between
# -5 and 5
'betawidth' : -10,
# Inverse Signal to noise ratio
'kappa' : 5,
'sub_iter_params': ['sparsity', 'cov_params']
}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################
