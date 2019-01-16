import numpy as np
# Paths
script_dir = '/global/homes/a/akumar25'
root_dir = '/global/homes/a/akumar25/uoicorr'
jobdir = 'est_comparison'

# Specify script to use:
script = 'est_comparison.py'

# List the set of arguments to the script(s) that will be iterated over
iter_params = {'sparsity' : np.linspace(0.2, 1, 10),
'correlations': [0.0, 0.2, 0.4, 0.6, 0.8, 1]}

# List arguments that will be held constant across all jobs:
comm_params = {'kappa' : 0.3, 'n_features' : 60, 'betadist' : ['\'uniform\''],
'block_size' : [6, 12, 20, 30],
'est_score': '\'r2\'', 'reps' : 5, 'selection_thres_mins' : [1.0],
'n_samples' : [ 60, 360,  660, 1020, 1320, 1680, 1980, 2340, 2640, 3000]}

# Parameters for ElasticNet
if script == 'elasticnet_block.py' or script == 'uoien_block.py':
	comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
	comm_params['n_alphas'] = 48

# Description of what the job is
desc = "Comparison of model performance using different estimation scores over a range of correlations,\
		sparsities, and sample sizes"

# Resources
job_time = '02:30:00'