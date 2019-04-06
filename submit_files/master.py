import numpy as np

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

###### Master list of parameters to be iterated over #######

exp_types =  ['UoILasso', 'UoIElasticNet', 'EN', 'CV_Lasso']

# Upon each run, the status of completed jobs will be compared
# to that required by this list.

iter_params = {

'betawidth' : [np.inf],

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
