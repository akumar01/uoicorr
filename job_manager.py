import numpy as np
import itertools


###### Master list of parameters to be iterated over #######

# Upon each run, the status of completed jobs will be compared
# to that required by this list.

iter_params = {

'betawidth' : [0, inf],

# Sparsity
'sparsity' : np.logspace(0.01, 1, 20),

# Linear interpolation strength
'interp_t': np.arange(0, 11, 10),

# Block sizes
'block_sizes' : [5, 10, 20, 50, 100],

# Block correlation
'correlation' : np.linspace(0, 0.8, 10),

# Exponential length scales
'L' : [1, 2.5, 5, 7.5, 10],

# n/p ratio #
'np_ratio': [0.5, 1, 2, 3, 5],

'exp_types': ['UoILasso', 'UoIElasticNet', 'EN', 'CV_Lasso']

}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'reg_params': [],
'n_models': 1,
'kappa' : 0.3, 
'n_features' : 500,
'est_score': 'r2',
'reps' : 10,
'stability_selection' : [1.0],
'n_boots_sel': 48,
'n_boots_est' : 48}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################

total_jobs = np.prod(len(val) for val in iter_params.values())

# Break up the total jobs into segments of 1000. Separate the algorithms 
# as some tend to run faster than others
