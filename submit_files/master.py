import numpy as np
import itertools
import pdb
from misc import get_cov_list

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

###### Master list of parameters to be iterated over #######

exp_types =  ['UoILasso', 'UoIElasticNet', 'EN', 'CV_Lasso']
# Estimated worst case run-time for a single repitition for each algorithm in exp_types 
algorithm_times = ['10:00:00', '10:00:00', '10:00:00', '10:00:00']

n_features = 500

# Block sizes
block_sizes = [25, 50, 100]

# Block correlation
correlation = [0, 0.08891397, 0.15811388, 0.28117066, 0.5]

# Exponential length scales
L = [10, 50, 200]

cov_list = get_cov_list(n_features, 90, correlation, block_sizes, L)

cov_params = [{'correlation' : t[0], 'block_size' : t[1], 'L' : t[2], 't': t[3]} for t in cov_list]

iter_params = {

'betawidth' : [0.1, 0.5, 2.5, np.inf],

'cov_params' : cov_params,

# Sparsity
'sparsity' : np.logspace(np.log10(0.05), 0, 10),

}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'cov_type' : 'interpolation',
'reg_params': [],
'n_models': 1,
'n_features' : n_features,
# n/p ratio #
'np_ratio': 5,
'est_score': 'r2',
'reps' : 1,
'stability_selection' : [1.0],
'n_boots_sel': 48,
'n_boots_est' : 48,
# Inverse Signal to noise ratio
'kappa' : np.linspace(0, 0.6, 5),
'sub_iter_params': ['kappa']}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################
