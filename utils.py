import numpy as np
from scipy.linalg import block_diag
import pdb
        
def gen_beta(n_features = 60, block_size = 6, sparsity = 0.6, betadist = 'uniform'):
    n_blocks = int(np.floor(n_features/block_size))
    
    n_nonzero_beta = int(sparsity * block_size)
    
    # Choose model coefficients to be uniformly distributed
    if betadist == 'uniform':
        beta = np.random.uniform(low=0, high=10, size=(n_features, 1))
    elif betadist == 'invexp':
        beta = invexp_dist(-5, 5, n_features)
        beta = np.reshape(beta, (n_features, 1))
    elif betadist == 'laplace':
        beta = np.random.laplace(scale = 1, size=(n_features, 1))
    elif betadist == 'clustered':
        beta = cluster_dist(low = 0, high = 25, n_clusters = 5, 
            cluster_width = 1, size = (n_features, 1))
    elif betadist == 'equal':
        beta = np.ones((n_features, 1))
    # Apply sparsity separately to each block
    mask = np.array([])
    for block in range(n_blocks):
        block_mask = np.zeros(block_size)
        block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
        np.random.shuffle(block_mask)
        mask = np.concatenate((mask, block_mask))
    mask = mask[..., np.newaxis]
    beta = beta * mask
    return beta

# New version of gen_beta that uses the betawidth parameter:
# A betawidth of 0 gives features that take on the same value
# A betawidth of inf is a uniform distribution on the range 0-10
def gen_beta2(n_features = 60, block_size = 6, sparsity = 0.6, betwidth = np.inf):

    # Handle 0 and np.inf as special cases
    if betawidth == np.inf:
        beta = np.random.uniform(low = 0, high = 10, size = (n_features, 1))
    elif betawidth == 0:
        beta = 5 * np.ones((n_features, 1))
    else:
        beta = np.random.laplace(scale = betawidth, loc = 5, size = (n_features, 1))

    # Apply sparsity separately to each block
    mask = np.array([])
    for block in range(n_blocks):
        block_mask = np.zeros(block_size)
        block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
        np.random.shuffle(block_mask)
        mask = np.concatenate((mask, block_mask))
    mask = mask[..., np.newaxis]
    beta = beta * mask
    return beta

# Generate coefficients that are of the same magnitude within each block
def equal_beta(low = 0, high = 10, n_features = 60, block_size = 6):

    n_blocks = int(np.floor(n_features/block_size))

    block_values = np.random.uniform(low, high, n_blocks)

    beta = np.zeros((n_features, 1))
    for i, block_value in enumerate(block_values):
        beta[block_size * i: block_size * (i + 1), 0] = block_value
    return beta

def interpolated_dist(lmbda, t, sigma, n_features = 60, low = -5, high = 5):
    # Return samples according to a mixture model of a Laplacian and Gaussian distribution. 
    # (1 - t) * Exp[-Abs[x]/lmbda] + t * Exp[-x^2/sigma]

    bad_sample =  lambda low, high, x: np.logical_or(x >= high, x <= low)

    beta_dist = lambda t, lmbda, sigma, x: (1 - t) * np.exp(-np.abs(x)/lmbda) + t * np.exp(-x**2/sigma)
    
    # Pure Laplacian case
    if t == 0:
        betas = np.random.laplace(scale = lmbda, size = (n_features, 1))

        # Check if any of the drawn betas lie outside the desired range:
        bad_idxs = np.argwhere(bad_sample(low, high, betas))

        # Switch out the samples that don't lie within the interval
        if bad_idxs.size > 0:
            for bad_idx in bad_idxs:
                x = np.random.laplace(scale = lmbda)
                while bad_sample(low, high, x):
                    x = np.random.laplace(scale = lmbda)
                betas[bad_idx] = x                

    # Pure Gaussian case
    elif t == 1:
        betas = np.random.uniform(scale = np.sqrt(sigma/2), size = (n_features, 1))

        # Check if any of the drawn betas lie outside the desired range:
        bad_idxs = np.argwhere(bad_sample(low, high, betas))

        # Switch out the samples that don't lie within the interval
        if len(bad_idxs) > 0:
            for bad_idx in bad_idxs:
                x = np.random.uniform(scale = np.sqrt(sigma/2))
                while bad_sample(low, high, x):
                    x = np.random.laplace(scale = np.sqrt(sigma/2))
                betas[bad_idx] = x                

    # Mixture: use rejection sampling:
    # Just use uniform distribution over the interval as the sampling function
    else: 
        betas = np.zeros(n_features)
        
        # Our desired distrbution will be bounded above by 1
        scale_factor = np.abs(high - low)

        for i in range(n_features):
            x = np.random.uniform(low, high)
            z = np.random.uniform(0, 1)
            y = beta_dist(t, lmbda, sigma, x)
            while z >= y:
                x = np.random.uniform(low, high)
                z = np.random.uniform(0, 1)
                y = beta_dist(t, lmbda, sigma, x)

            betas[i] = x
    
    return betas 



# Generate toy data to fit given number of features, covariance structure, signal to noise, and sparsity
# Note that sparsity is applied within blocks. When this is not desired, set the block size to equal n_features
def gen_data(n_samples = 5 * 60, n_features = 60, kappa = 0.3,
            covariance = np.diag(np.ones(60)), beta = np.random.uniform(0, 10, (60, 1))):

    # draw samples from a multivariate normal distribution cenetered around 0
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)
    X_test = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)

    # signal and noise variance
    signal_variance = np.sum(covariance * np.dot(beta, beta.T))
    noise_variance = kappa * signal_variance

    # draw noise
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
    noise_test = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))

    # response variable
    y = np.dot(X, beta) + noise
    y_test = np.dot(X_test, beta) + noise_test

#     # Center response
#     y = y - np.mean(y)
#     y_test = y_test - np.mean(y_test)


    return X, X_test, y, y_test

# Return covariance matrix based on covariance type:
def gen_covariance(cov_type, n_features = 60, block_size = 6, **kwargs):
    if cov_type == 'block':
        return block_covariance(n_features, block_size, **kwargs)
    elif cov_type == 'falloff':
        return exp_falloff(n_features, **kwargs)

# Create a block diagonal covariance matrix 
def block_covariance(n_features = 60, block_size = 6, correlation = 0):

    n_blocks = int(n_features/block_size)

    # create covariance matrix for block
    block_sigma = correlation * np.ones((block_size, block_size)) 
    np.fill_diagonal(block_sigma, np.ones(block_size))
    # populate entire covariance matrix
    rep_block_sigma = [block_sigma] * n_blocks
    sigma = block_diag(*rep_block_sigma)
    return sigma

# Create a covariance matrix where the correlations are given by an exponential
# fall off
def exp_falloff(n_features = 60, block_size = None, L = 1):
    indices = np.arange(n_features)
    distances = np.abs(np.subtract.outer(indices, indices))
    sigma = np.exp(-distances/L)
    return sigma

def interpolate_covariance(cov_type1, cov_type2, interp_coeffs = np.linspace(0, 1, 11),
    n_features = 60, block_size = 6, cov_type1_args = {}, cov_type2_args = {}):
    # Start from covariance matrix 1
    cov_type1 = globals()[cov_type1]
    cov_type2 = globals()[cov_type2]
    cov_0 = cov_type1(n_features, block_size, **cov_type1_args)
    cov_n = cov_type2(n_features, block_size, **cov_type2_args)
    cov = []
    
    for t in interp_coeffs:
        sigma = (1 - t) * cov_0 + t * cov_n
        sigma = sigma.tolist()
        cov.append({'t': t, 'sigma': sigma})
    # Return as nested lists to be saved as .json

    return cov

# Sample from 1/Exp[-Abs|x|], properly normalized
def invexp_dist(low, high, n_samples):

    x = np.linspace(low, high, 10000)
    fx = np.exp(np.abs(x))

    # normalize
    fx = fx/np.sum(fx)

    # CDF
    Fx = np.cumsum(fx)

    # generate uniform random variables
    u = np.random.uniform(size = n_samples)

    # Apply the inverse CDF
    y = np.array([])
    for ux in u:
            y = np.append(y, x[np.argwhere(Fx == min(Fx[(Fx - ux) > 0])).ravel()])

    return y

# Return samples clustered around a set of points
def cluster_dist(low, high, n_clusters, cluster_width, size):

    x = np.zeros(size)

    # Evenly space clusters
    cluster_centers = np.cumsum((high - low)//n_clusters * np.ones(n_clusters))

    for idx in range(size[0]):
        # Randomly choose cluster center
        center = np.random.choice(cluster_centers)
        # Sample from uniform distribution centered on cluster with width
        # given by cluster_width                                                
        x[idx] = np.random.uniform(low = center - cluster_width/2, 
            high = center + cluster_width/2)

    return x

# Standardize calculation of FNR, FPR, and Selection Accuracy
# threshold : Set things smaller than 1e-6 explicitly to 0 in beta_hat
def FNR(beta, beta_hat, threshold = False):
    beta, beta_hat = tile_beta(beta, beta_hat)
    if threshold:
        beta_hat[beta_hat < 1e-6] = 0
    false_negative_rate = np.zeros(beta_hat.shape[0])
    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()
        try:
            false_negative_rate[i] = np.count_nonzero(b[(bhat == 0).ravel()])\
            /(np.count_nonzero(b))
        except ZeroDivisionError:
            print('Shit!')

    return false_negative_rate

def FPR(beta, beta_hat, threshold = False):

    beta, beta_hat = tile_beta(beta, beta_hat)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    false_positive_rate = np.zeros(beta_hat.shape[0])
    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()
        try:
            false_positive_rate[i] = np.count_nonzero(bhat[(b == 0).ravel()])\
                                /(np.where(b == 0)[0].size)
        except ZeroDivisionError:
            # No sparsity case
            false_positive_rate[i] = 0

    return false_positive_rate


def selection_accuracy(beta, beta_hat, threshold = False):

    beta, beta_hat = tile_beta(beta, beta_hat)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    selection_accuracy = np.zeros(beta_hat.shape[0])
    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()
        selection_accuracy[i] = 1 - \
        np.count_nonzero(1 * np.logical_xor(bhat != 0, b != 0))\
        /(bhat.size + b.size)
    return selection_accuracy

def estimation_error(beta, beta_hat, threshold = False):        
    beta, beta_hat = tile_beta(beta, beta_hat)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    ee = np.zeros(beta_hat.shape[0])
    median_ee = np.zeros(beta_hat.shape[0])
    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()
        p = b.size
        median_ee[i] = np.median(np.sqrt(np.power(b - bhat, 2)))
        ee[i] = 1/p * np.sqrt(np.sum(np.power(b - bhat, 2)))

    return ee, median_ee


def tile_beta(beta, beta_hat):

    if np.ndim(beta_hat) == 1:
        beta_hat = beta_hat[np.newaxis, :]

    if np.ndim(beta) == 1:
        beta = beta[np.newaxis, :]

    if beta.shape != beta_hat.shape: 
        beta = np.tile(beta, [int(beta_hat.shape[0]/beta.shape[0]), 1])

    return beta, beta_hat
