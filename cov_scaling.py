import numpy as np


# Correlate regression performance with statistical measures of the graph formed by the covriance
# matrix of covariates


# Return the weighted degree distribution of the covariance matrix
def weighted_degree_distribution(cov):
	# Exclude the diagonal
	cov = cov - np.diag(np.diag(cov))

	ddist = np.zeros(cov.shape[0])
	for i in range(cov.shape[0]):
		ddist[i] = np.sum(cov[i, :])
	return ddist 

# Return the weighted clustering coefficient of the covariance matrix
# def weighted_path_length(cov):
# 	cov = cov - np.diag(np.diag(cov))

# 	pthlgth = np.zeros(cov.shape[0]):
# 	for i in range(cov.shape[0]):		