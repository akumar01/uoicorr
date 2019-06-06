import numpy as np 
from scipy.integrate import quad


# Exact model selection relevant portion of the KL divergence
def calc_KL_div(mu_hat, sigma_hat, sigma):

	# mu_hat: vector of X * \beta terms estiamted from the data

	n = mu_hat.size

	exact_KL_div = \
	1/2 * (n * np.log(sigma_hat**2)/np.sqrt(2 * np.pi * sigma**2) + \
		   n * sigma**2/(sigma_hat**2) + 1/(sigma_hat**2) * np.linalg.norm(mu_hat)**2)

	return exact_KL_div

# Calculate the AIC estimate of the KL div term above
def AIC(y_true, y_pred, n_features):

	n = y_true.size

	AIC =1/2 * (np.log(2 * np.pi/n * np.linalg.norm(y_true - y_pred)**2) + 1) + n_features 

	return AIC
