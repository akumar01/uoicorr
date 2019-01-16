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

# Create a block diagonal covariance matrix 
def block_covariance(n_features = 60, correlation = 1, block_size = 6):

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
def exp_falloff(n_features = 60, L = 1):
    indices = np.arange(n_features)
    distances = np.abs(np.subtract.outer(indices, indices))
    sigma = np.exp(-distances/L)
    return sigma

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