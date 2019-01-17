import numpy as np

# Paths
script_dir = '/global/homes/a/akumar25'
root_dir = '/global/homes/a/akumar25/uoicorr'
jobdir = '01162019'

# Specify script to use:
script = 'uoicorr_base.py'

# List the set of arguments to the script(s) that will be iterated over
iter_params = {
'sparsity' : [0.2, 0.4, 0.6, 0.8, 1],
'exp_type' : ['UoILasso', 'UoIElasticNet', 'EN']
}

# List arguments that will be held constant across all jobs:
comm_params = {
'kappa' : 0.3, 
'n_features' : 60,
'betadist' : 'uniform',
'block_size' : 60,
'est_score': 'r2',
'reps' : 50,
'selection_thres_mins' : [1.0],
'cov_type' : 'falloff',
'cov_params':  [{'L' : x} for x in [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
'n_samples' : 300}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

# Description of what the job is
desc = "Exponentially falling off correlations"

# Resources
job_time = '02:30:00'