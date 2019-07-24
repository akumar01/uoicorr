import numpy as np
from pyuoi.utils import log_likelihood_glm
import scipy
import pdb

# Generalized information criterion with arbitrary penalty
def GIC(y, y_pred, model_size, penalty):

    y = y.ravel()
    y_pred = y_pred.ravel()

    ll = log_likelihood_glm('normal', y, y_pred)
    return -2 * ll + penalty * model_size

# Extended BIC (found in literature)
def eBIC(y, y_pred, n_features, model_size):

    n_samples = y.size

    # Kappa is a hyperparameter tuning the strength of the effect (1 -> no effect)
    kappa = 0.5

    eBIC = GIC(y, y_pred, model_size, np.log(n_samples)) + \
    n_features *  2 * (1 - kappa) * np.log(float(scipy.special.binom(n_features, model_size)))

    return eBIC

# modified BIC penalty with some prior on the model size
def mBIC(y, y_pred, model_size, sparsity_prior):

    # make sure sparsity prior is epsilon less than 1
    if sparsity_prior == 1:
        sparsity_prior = 0.9999

    mBIC =  BIC(y, y_pred, model_size) + 2 * model_size * np.log(1/sparsity_prior - 1)

    return mBIC

# Full Bayesian model penalty selection
def bayesian_lambda_selection(y, y_pred, n_features, model_size, sparsity_prior, penalty):

    y = y.ravel()
    y_pred = y_pred.ravel()

    n_samples = y.size

    # Log likelihood
    ll = log_likelihood_glm('normal', y, y_pred)

    # Regularization Penalty (prior)
    p1 = 2 * penalty * model_size

    # Normal BIC penalty
    BIC = model_size * np.log(n_samples)

    # Second order Bayes factor approximation
    RSS = np.sum((y - y_pred)**2)
    BIC2 = n_samples**3/(2 * RSS*3)

    # Term arising from normalization
    BIC3 = model_size * np.log(2 * np.pi)

    # If provided with a list of sparsity estimates, we are specifying
    # a beta hyperprior, and need to integrate over it correspondingly
    if not np.isscalar(sparsity_prior):
        M_k = beta_binomial_model(sparsity_prior, n_features, model_size)
    else:
        if sparsity_prior == 1:
            sparsity_prior = 0.999

        # Model probability prior
        M_k = scipy.special.binom(n_features, model_size) * \
              sparsity_prior**model_size * (1 - sparsity_prior)**(n_features - model_size)

    P_M = 2 * np.log(M_k)

#    bayes_factor = 2 * ll - BIC - BIC2 + BIC3 - p1 + P_M

    return ll, p1, BIC, BIC2, BIC3, M_k, P_M


# Return a posterior estimate for the binomial parameter given a 
# beta-distribution prior on estimates of that parameter 
def beta_binomial_model(x, n, k):

    # drop all entries that are 0 
    x = x[x != 0]



    # Fit the parameters of the beta distribution
    a, b, _, _ = scipy.stats.beta.fit(x, floc = 0, fscale = 1)

    p = scipy.special.binom(n, k) * \
            scipy.special.beta(k + a, n - k + b)/scipy.special.beta(a, b)
    
    return p
