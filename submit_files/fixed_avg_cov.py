import numpy as np
import sys
sys.path.append('..')
from utils import cov_spread

script_dir = '/global/homes/a/akumar25/repos/uoicorr'

###### Master list of parameters to be iterated over #######

exp_types =  ['UoILasso', 'UoIElasticNet', 'EN', 'CV_Lasso']

# Time to request 
algorithm_times = ['04:00:00', '04:00:00', '04:00:00', '04:00:00']

# Generate sigma matrices of fixed average correlation
cov_set = [list(cov_spread(np.linspace(0.025, 0.35, 2), cov_type)) 
				for cov_type in ['block', 'exp_falloff', 'interpolate', 'random']]

# Flatten the list
flattened_cov_set = [sigmas for sigma_type in cov_set for sigmas in sigma_type]

iter_params = {

'cov_params' :  flattened_cov_set,
'sparsity' : list(np.logspace(0.05, 0.4, num=5)),

}

#############################################################

##### Common parameters held fixed across all jobs ##########
comm_params = {
'cov_type' : 'fixed_avg',
'n_features' : 500,
'n_samples' : 2000,
'block_size' : 1000,
'betawidth' : 'uniform',
'reg_params': [],
'n_models': 1,
'kappa' : 0.3, 
'est_score': 'r2',
'reps' : 1,
'stability_selection' : [1.0],
'n_boots_sel': 48,
'n_boots_est' : 48}

# Parameters that will be internally iterated in each job.
comm_params['sub_iter_params'] = ['cov_params']

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

###############################################################
