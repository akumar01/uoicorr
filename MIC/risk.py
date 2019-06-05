import numpy as np 
from scipy.integrate import quad


# Exact model selection relevant portion of the KL divergence
def calc_KL_div(mu_hat, sigma_hat, sigma):

	# mu_hat: vector of X * \beta terms estiamted from the data

	n = mu_hat.size

	exact_KL_div = n/2 * (np.log(sigma_hat**2) + sigma**2/(sigma_hat**2)) + 1/(2 * sigma_hat**2) * np.linalg.norm(mu_hat)**2

	return exact_KL_div