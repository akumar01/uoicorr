import numpy as np

# Categorize results according to selection method
def init_results_container(selection_methods, fields, num_tasks, n_features):

	# Use a nested dict
	results = {selection_method: {} for selection_method in selection_methods}

	# # For each selection method, record the following quantities:
	# fields = ['FNR', 'FPR', 'sa', 'ee', 'median_ee', 'r2', 'beta_hats', 
	# 		  'MSE', 'AIC', 'BIC']

	for selection_method in selection_methods:
		# For all except beta_hat, initialize arrays of size num_tasks
		for field in Diff(fields, 'beta_hats'):
			results[selection_method][field] = np.zeros(num_tasks)
		results[selection_method]['beta_hats'] = np.zeros((num_tasks, n_features))

	return results

def gather_results