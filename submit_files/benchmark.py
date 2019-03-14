import numpy as np

# Paths
script_dir = '/global/homes/a/akumar25'
root_dir = '/global/homes/a/akumar25/uoicorr'
jobdir = '03142019'

# Specify script to use:
script = 'uoicorr_base.py'

# List the set of arguments to the script(s) that will be iterated over
iter_params = {
'sparsity' : [0.01, 0.5, 1],
'exp_type' : ['UoILasso', 'UoIElasticNet', 'EN'],
'cov_params':  [{'correlation' : x} for x in [0, 0.4, 0.8]],
'block_size' : [5, 100],
'n_samples' : [500, 3000]
}

# List arguments that will be held constant across all jobs:
comm_params = {
'kappa' : 0.3, 
'n_features' : 1000,
'betadist' : 'uniform',
'est_score': 'r2',
'reps' : 10,
'selection_thres_mins' : [1.0],
'cov_type' : 'block'}

# Parameters for ElasticNet
comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
comm_params['n_alphas'] = 48

# Description of what the job is
desc = "Benchmarking on large feature set to determine the statistics of runtimes for repetitions"

# Resources
job_time = '12:00:00'