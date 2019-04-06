import numpy as np
from utils import cov_sweep

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

###### Master list of parameters to be iterated over #######

exp_types =  ['UoILasso', 'UoIElasticNet', 'EN', 'CV_Lasso']

# Time to request 
algorithm_times = ['10:00:00', '10:00:00', '10:00:00', '10:00:00']

# Upon each run, the status of completed jobs will be compared
# to that required by this list.

iter_params = {

# Iterate through each of 25 different average correlations
# Need to flatten this list!!
'cov_params' : [cov_sweep(np.linspace(0.025, 0.35, 25), cov_type) 
				for cov_type in ['block', 'exp_falloff', 'interpolate', 'random']]
}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'sparsity' : list(np.logspace(0.05, 0.4, num=5))
'cov_type' : 'fixed_avg',
'n_features' : 1000,
'n_samples' : 2000,
'block_size' : 1000,
'betawidth' : 'uniform',
'reg_params': [],
'n_models': 1,
'kappa' : 0.3, 
'n_features' : 500,
'est_score': 'r2',
'reps' : 10,
'stability_selection' : [1.0],
'n_boots_sel': 48,
'n_boots_est' : 48}

# Parameters that will be internally iterated in each job.
comm_params['sub_iter_params'] = ['sparsity', 'cov_params']

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################
