import numpy as np
import pdb

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

# Return the weighted clustering coefficient of the covariance matrix
def weighted_clustering_coefficient1(cov):

	cov = cov - np.diag(np.diag(cov))

	cdist = np.zeros(cov.shape[0])
	for i in range(cov.shape[0]):
		# Compute the number of immediate neighbors of node i
		v = np.count_nonzero(cov[i, :])
		# Numerator is the 'total weight of relationship surrounding node 
		numerator = 0
		neighborhood = np.where(cov[i, :] != 0)
		for j in neighborhood[0]:
			numerator += np.sum(cov[j, neighborhood])
		try:
			cdist[i] = numerator/(v * (v - 1))
		except ZeroDivisionError:
			cdist[i] = np.nan
	return cdist


