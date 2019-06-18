import numpy as np
from scipy.integrate import quad
import pdb
from pyuoi.utils import log_likelihood_glm
from pyuoi.utils import AIC as AIC_

from utils import gen_data

# Exact model selection relevant portion of the KL divergence
def calc_KL_div(mu_hat, sigma_hat, sigma):

	# mu_hat: vector of X * \beta terms estiamted from the data

	n = mu_hat.size

	mu_hat = mu_hat.ravel()

	exact_KL_div = \
	1/2 * (n * np.log(2 * np.pi * sigma**2) + \
		   n * sigma_hat**2/(sigma**2) + 1/(sigma**2) * np.sum(mu_hat**2))


	return exact_KL_div

# Exact risk function with penalty term
def exact_penalized_KL(beta_hat, penalty, mu_hat, sigma_hat, sigma):

	eKD = calc_KL_div(mu_hat, sigma_hat, sigma)
	penalized_KL = eKD - penalty * np.linalg.norm(beta_hat, 0)

	return penalized_KL

# Use MC to estimate the KL div
def MC_KL_estimate(mu_hat, sigma_hat, ss, covariance, beta):

	mu_hat = mu_hat.ravel()

	n_features = beta.size
	n_samples = mu_hat.size
	n_MC_batches = 1000
	MC_KL_div = 0
	for n in range(n_MC_batches):
		# generate data from the true model:
		x = np.random.multivariate_normal(mean = np.zeros(n_features), cov = covariance, size = n_samples)
		noise = np.random.normal(loc = 0, scale = np.sqrt(ss), size = (n_samples, 1))
		y =  x @ beta + noise

		y = y.ravel()
		MC_KL_div += 1/n_MC_batches * empirical_KL_estimate(y, mu_hat)

	return MC_KL_div

# Now, use the data used to fit the model to also estimate the KL_div. This should reveal a systematic bias
def empirical_KL_estimate(y, mu_hat):

	n = y.size

	y = y.ravel()
	mu_hat = mu_hat.ravel()

	empirical_KL_div = n/2 * (np.log(2 * np.pi) + 1 + np.log(np.mean((y - mu_hat)**2)))

	return empirical_KL_div

# Calculate the AIC estimate of the KL div term above. AIC contains bias correction
def AIC(y_true, mu_hat, sigma_hat, n_features):

	n_samples = y_true.size

	eKLe = empirical_KL_estimate(y_true, mu_hat, sigma_hat)

	AIC = eKLe + n_features

	return AIC

# Manually specify a model complexity penalty.
def MIC(y_true, mu_hat, k, penalty):

	y_true = y_true.ravel()
	mu_hat = mu_hat.ravel()

	eKLe = empirical_KL_estimate(y_true, mu_hat)

	MIC = eKLe + penalty * k

	return MIC
