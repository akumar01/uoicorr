import numpy as np
from scipy.linalg import block_diag

def gen_data(n_features = 60, block_size = 6, kappa = 0.1,  covariance = np.diag(np.ones(60)), sparsity = 0.6):

    n_samples = 5 * n_features

    n_blocks = int(np.floor(n_features/block_size))
    
    n_nonzero_beta = int(sparsity * block_size)
    
    # Choose model coefficients to be uniformly distributed
    beta = np.random.uniform(low=0, high=10, size=(n_features, 1))

    # Apply sparsity separately to each block
    mask = np.array([])
    for block in range(n_blocks):
        block_mask = np.zeros(block_size)
        block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
        np.random.shuffle(block_mask)
        mask = np.concatenate((mask, block_mask))
    mask = mask[..., np.newaxis]
    beta = beta * mask
    
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


    return X, X_test, y, y_test, beta