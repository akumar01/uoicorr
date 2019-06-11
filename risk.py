import numpy as np 
from scipy.integrate import quad
import pdb
from pyuoi.utils import log_likelihood_glm
from pyuoi.utils import AIC as AIC_


# Exact model selection relevant portion of the KL divergence
def calc_KL_div(mu_hat, sigma_hat, sigma):

	# mu_hat: vector of X * \beta terms estiamted from the data

	n = mu_hat.size

	exact_KL_div = \
	1/2 * (n * np.log(2 * np.pi * sigma_hat**2) + \
		   n * sigma**2/(sigma_hat**2) + 1/(sigma_hat**2) * np.linalg.norm(mu_hat)**2)

	return exact_KL_div

# Use MC to estimate the KL div
def MC_KL_estimate(mu_hat, sigma_hat, sigma):
	n = mu_hat.size
	n_samples = n
	# Draw y from the true distribution to evaluate the expectation
	y = np.random.normal(0, sigma, size = n_samples)

	MC_KL_div = 0

	MC_KL_div = np.array([np.array([np.log(1/np.sqrt(2 * np.pi * sigma_hat**2) * np.exp(-(y[i] - mu_hat[j])**2/(2 * sigma_hat**2))) for j in range(n)]) for i in range(n_samples)])
	MC_KL_div = np.mean(np.sum(MC_KL_div, axis = 1))

	return -1 * MC_KL_div

# Now, use the data used to fit the model to also estimate the KL_div. This should reveal a systematic bias
def empirical_KL_estimate(y, mu_hat, sigma_hat):

	n = y.size

	empirical_KL_div = 1/2 * (n * np.log(2 * np.pi) - n * np.log(n) + n + 2 * np.log(y - mu_hat))

	return -1 * empirical_KL_div

# Calculate the AIC estimate of the KL div term above. AIC contains bias correction
def AIC(y_true, mu_hat, sigma_hat, n_features):

	n_samples = y_true.size

	eKLe = empirical_KL_estimate(y_true, mu_hat, sigma_hat)

	AIC = eKLe + n_features

	return AIC

# Manually specify a model complexity penalty.
def MIC(y_true, mu_hat, sigma_hat, n_features, penalty):

	n_samples = y_true.size

	eKLe = empirical_KL_estimate(y_true, mu_hat, sigma_hat)

	MIC = eKLe + penalty * n_features

	return MIC