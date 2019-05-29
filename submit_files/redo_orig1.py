import numpy as np
import itertools
import pdb
from misc import get_cov_list

script_dir = '/global/homes/a/akumar25/repos/uoicorr'


desc = 'Redo the original 60 feature simulations'

exp_types =  ['UoILasso', 'UoIElasticNet', 'CV_Lasso', 'EN']
# Estimated worst case run-time for a single repitition for each algorithm in exp_types 
algorithm_times = ['8:00:00']

n_features = 60

# Block sizes
block_sizes = [6, 12, 20, 30]

# Block correlation
correlation = [0, 0.2 0.4, 0.6, 0.8, 1]

# Exponential length scales
L = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

cov_list, _ = get_cov_list(n_features, 288, correlation, block_sizes, L, n_supplement = 0)


cov_params = [{'correlation' : t[0], 'block_size' : t[1], 'L' : t[2], 't': t[3]} for t in cov_list]

# Interpolation
t = np.linspace(0, 1, 11)

interp_params = [{'correlation' : 0.8, 'block_size' : 6, 'L' : 10, 't' : tt} for tt in t]

cov_params.extend(interp_params)

iter_params = {
'cov_params' : cov_params
}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'sparsity' : [0.2, 0.4, 0.6, 0.8, 1],
'cov_type' : 'interpolation',
'reg_params': [],
'n_models': 1,
'n_features' : n_features,
# n/p ratio #
'np_ratio': 5,
'est_score': 'r2',
'reps' : 50,
'stability_selection' : 1.,
'n_boots_sel': 48,
'n_boots_est' : 48,
'betawidth' : np.inf,
# Inverse Signal to noise ratio
'kappa' : 1./0.3,
'sub_iter_params': {}
}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################
