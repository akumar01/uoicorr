import numpy as np
from scipy.linalg import block_diag
from misc import *
import pdb
import traceback
import time
        
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
def gen_beta2(n_features = 60, block_size = 10, sparsity = 0.6, 
            betawidth = np.inf, sparsity_profile = 'uniform', 
            n_active_blocks = None, seed = None):
    n_blocks = int(np.floor(n_features/block_size))

    n_nonzero_beta = int(sparsity * block_size)
    # Repeatable coefficients
    if seed is not None:
        np.random.seed(int(seed))

    # Handle 0, np.inf, and < 0 (inverted exponential) as special cases
    if betawidth == np.inf:
        beta = np.random.uniform(low = 0, high = 10, size = (n_features, 1))
    elif betawidth == 0:
        beta = 5 * np.ones((n_features, 1))
    elif betawidth < 0:
        beta = invexp_dist(-5, 5, n_features)
    else:
        beta = np.zeros((0,))    # empty for now
        while beta.shape[0] < n_features: 
            b = np.random.laplace(scale = betawidth, loc = 5, size=(n_features,))
            accepted = b[(b >= 0) & (b <= 10)]
            beta = np.concatenate((beta, accepted), axis=0)
        beta = beta[:n_features] 
        beta = beta[:, np.newaxis]   # we probably got more than needed, so discard extra ones

    if sparsity_profile == 'uniform':
        # Apply sparsity uniformly across blocks
        mask = np.array([])
        for block in range(n_blocks):
            block_mask = np.zeros(block_size)
            block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
            np.random.shuffle(block_mask)
            mask = np.concatenate((mask, block_mask))
    elif sparsity_profile == 'block':
        # Choose blocks to be either all active or all inactive
        n_active_blocks = np.int(_nonzero_beta/block_size)
        active_blocks = np.random.choice(np.arange(n_blocks))
        mask = np.array([])
        for block in range(n_blocks):
            if block in active_blocks:
                block_mask = np.ones(block_size)
            else:
                block_mask = np.zeros(block_size)
            mask = np.concatenate((mask, block_mask))

    elif sparsity_profile == 'block_sparse':
        # Choose blocks to be all active or all inactive, and then 
        # apply sparsity uniformly within blocks
        if n_active_blocks is None:
            print('Need n_active_blocks!\n')
        active_blocks = np.random.choice(np.arange(n_blocks))
        mask = np.array([])
        for block in range(n_blocks):
            if block in active_blocks:
                block_mask = np.ones(block_size)
                block_mask[:n_nonzero_beta] = np.ones(n_nonzero_beta)
                np.random.shuffle(block_mask)
            else:
                block_mask = np.zeros(block_size)
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
def gen_data(n_samples = 5 * 60, n_features = 60, kappa = 3,
            covariance = np.diag(np.ones(60)), beta = np.random.uniform(0, 10, (60, 1)), seed = None):

    # For consistency across different runs
    if seed is not None:
        np.random.seed(int(seed))
        
    # draw samples from a multivariate normal distribution cenetered around 0
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)
    X_test = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)


    # Kappa is the signal to noise ratio

    signal = np.var(X @ beta)
    noise_variance = signal/kappa

    # draw noise
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))
    
    signal = np.var(X_test @ beta)
    noise_variance = signal/kappa

    noise_test = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=(n_samples, 1))

    # response variable
    y = np.dot(X, beta) + noise
    y_test = np.dot(X_test, beta) + noise_test
#     # Center response
#     y = y - np.mean(y)
#     y_test = y_test - np.mean(y_test)

    return X, X_test, y, y_test, noise_variance

# Given a vector of desired average correlations, return a set of covariance
# matrices from each of the block, exp, and interpolated classes
# num: number of entries (at most) to return per avg_cov
def cov_spread(avg_covs, cov_type, num, n_features=1000):
    sigmas = []

    # Filter possible block sizes given n_features
    block_sizes = np.arange(5, int(n_features/2) + 2)
    block_sizes = np.array([b for b in block_sizes if not np.mod(n_features, b)])
    
    for i, avg_cov in enumerate(avg_covs):
        start = time.time()
        sigmas.append([])        

        if cov_type == 'block':

            for block_size in block_sizes:
                try:
                    ss = gen_avg_covariance('block', avg_cov, n_features, block_size=block_size)
                    print(ss.shape)
                    sigmas[i].append({'sigma': ss,
                                        'avg_cov': avg_cov, 'cov_type': cov_type})
                except:
                    pass
                    #traceback.print_exc()

#             # Block diagonal matrix, iterate correlation strength
#             corr = np.linspace(0.05, 0.5, 25)
#             for c in corr:
#                 try:
#                     ss = gen_avg_covariance('block', avg_cov, n_features, correlation=c)
#                     print(ss.shape)
#                     sigmas[i].append({'sigma': ss,
#                                         'avg_cov': avg_cov, 'cov_type': cov_type})
#                 except:
#                     pass
#                     #traceback.print_exc()

        elif cov_type == 'exp_falloff':
            # Exponential correlation matrix
            try:
                ss = gen_avg_covariance('exp_falloff', avg_cov, n_features) 
                print(ss.shape)
                sigmas[i].append({'sigma': ss,
                                    'avg_cov': avg_cov, 'cov_type': cov_type})
            except:
                pass
                #traceback.print_exc()

        elif cov_type == 'interpolate':
        
            # Interpolation
            # Interpolate between a sets of block_diagonal and exponential matrices
            L = np.linspace(1, n_features, 10)        
            corr = np.linspace(0.05, 0.5, 10)
            block_covs = []
            for block_size in block_sizes:
                for c in corr:
                    block_covs.append({'block_size': block_size, 'correlation': c})

            exp_covs = [{'L': ll} for ll in L]

            for bc in block_covs:
                for ec in exp_covs:
                    try:
                        ss = gen_avg_covariance('interpolate', avg_cov = avg_cov, n_features = n_features,
                            cov_type1='block_covariance', cov_type2='exp_falloff', cov_type1_args=bc, 
                            cov_type2_args=ec)
                        print(ss.shape)
                        sigmas[i].append({'sigma': ss,
                            'avg_cov': avg_cov, 'cov_type': cov_type}) 
                    except:
                        pass

        elif cov_type == 'random':
            # Random covariance matrix with varying degrees of sparsity

            # Subtract off diagonal contribution
            residual_correlation = n_features**2 * avg_cov - n_features 
            if residual_correlation < 0:
                raise Exception('The desired avg_cov cannot be achieved by a random matrix of this feature size')

            sparsities = np.logspace(-2, 0, 25)
            # For each sparsity, generate random numbers bounded between
            # 0 and 1 and rescale them uniformly so that they give the
            # desired average correlation
            for sidx, s in enumerate(sparsities):
                num_nonzero = int(s * ( n_features**2 - n_features))

                entries = np.random.uniform(size = num_nonzero)

                entries *= residual_correlation/(sum(entries))

                # Distribute the entries randomly in off-diagonal locations
                idx = np.array([np.unravel_index(ii, (n_features, n_features)) 
                                    for ii in np.arange(n_features**2)])

                offdiagidx = idx[[ii[0] != ii[1] for ii in idx]]
                # Where to put the entries off-diagonal
                locs = np.random.choice(len(offdiagidx), len(entries), replace=False)
                locsidx = (np.array(offdiagidx[locs])[:, 0], np.array(offdiagidx[locs])[:, 1])
                Sigma = np.identity(n_features)

                Sigma[locsidx[0], locsidx[1]] = entries

                sigmas[i].append({'sigma': Sigma, 'avg_cov': avg_cov, 'cov_type': cov_type})
                print('sidx = %d' % sidx)

        print('Iteration time: %f' % (time.time() - start))

    # Return a (roughly) consistent number of covariance matrices for each avg_cov

    filtered_sigmas = []
    for i in range(len(avg_covs)):
        num_take = min(len(sigmas[i]), num)
        # Randomly select num_take out of the entries in sigma
        take = np.random.choice(len(sigmas[i]), num_take, replace=False)
        filtered_sigmas.append([sigmas[i][t] for t in take])

    return filtered_sigmas


# Return covariance matrix based on desired average correlation
def gen_avg_covariance(cov_type, avg_cov = 0.1, n_features = 60, **kwargs):

    if cov_type == 'block':
        # Two free parameters here: Solve for whichever one is not
        # specified
        if 'block_size' in kwargs:
            block_size = kwargs['block_size']
            correlation = avg_cov * n_features/block_size
            if correlation >=1:
                raise Exception('Desired average correlation is incompatible\
                    with block structure')
        elif 'correlation' in kwargs:
            correlation = kwargs['correlation']
            block_size = int(n_features * avg_cov/correlation)
            if block_size > n_features or block_size < 1:
                raise Exception('Desired average correlation is incompatible with\
                    block structure')
        return block_covariance(n_features, 
            block_size = block_size, correlation = correlation)
    elif cov_type == 'exp_falloff':
        L = solve_L(n_features, avg_cov)
        print(L)
        return exp_falloff(n_features, L = L)
    elif cov_type == 'interpolate':
        cov_type1 = kwargs['cov_type1']
        cov_type2 = kwargs['cov_type2']
        cov_type1 = globals()[cov_type1]
        cov_type2 = globals()[cov_type2]
        cov_type1_args = kwargs['cov_type1_args']
        cov_type2_args = kwargs['cov_type2_args']
        cov_1 = cov_type1(n_features, **cov_type1_args)
        cov_2 = cov_type2(n_features, **cov_type2_args)

        t = solve_t(avg_cov, cov_1, cov_2)

        return interpolate_covariance(kwargs['cov_type1'], kwargs['cov_type2'],
                                [t], n_features, cov_type1_args, cov_type2_args)[0]['sigma']

    else:
       raise Exception('invalid or missing cov_type')

# # Return covariance matrix based on covariance type:
# def gen_covariance(cov_type, n_features = 60, **kwargs):
#     if cov_type == 'block':
#         return block_covariance(n_features, **kwargs)
#     elif cov_type == 'falloff':
#         return exp_falloff(n_features, **kwargs)

# Return covariance matrix given the 5 necessary parameters
def gen_covariance(n_features, correlation, block_size, L, t, threshold = 0):
    s0 = block_covariance(n_features, correlation, block_size)
    s1 = exp_falloff(n_features, L)
    s = (1 - t) * s0 + t * s1
    if threshold > 0: 
        s[s < threshold] = 0
    return s

# Create a block diagonal covariance matrix 
def block_covariance(n_features = 60, correlation = 0, block_size = 6):

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

def interpolate_covariance(interp_coeffs = np.linspace(0, 1, 11),
    n_features = 60, block_args = {}, falloff_args = {}):
    # Start from covariance matrix 1
    cov_0 = block_covariance(n_features, **cov_type1_args)
    cov_n = exp_falloff(n_features, **cov_type2_args)
    cov = []
    
    if not hasattr(interp_coeffs, '__len__'):
        interp_coeffs = [interp_coeffs]

    for t in interp_coeffs:
        sigma = (1 - t) * np.array(cov_0) + t * np.array(cov_n)
    #    sigma = sigma.tolist()
        cov.append({'t': t, 'sigma': sigma})
    # Return as nested lists to be saved as .json

    return cov

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

# Calculate estimation error
# Do so using only the overlap of the estimated and true support sets
def estimation_error(beta, beta_hat, threshold = False):        
    beta, beta_hat = tile_beta(beta, beta_hat)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    ee = np.zeros(beta_hat.shape[0])
    median_ee = np.zeros(beta_hat.shape[0])

    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()

        common_support = np.bitwise_and(b != 0, bhat != 0)
        p = bhat[common_support].size
        if p > 0:
            median_ee[i] = np.median(np.sqrt(np.power(b[common_support] - \
                                        bhat[common_support], 2)))
            ee[i] = 1/p * np.sqrt(np.sum(np.power(b[common_support] - \
                                  bhat[common_support], 2)))
        else:
            median_ee[i] = np.nan
            ee[i] = np.nan

    return ee, median_ee


# Calculate the estimation error, separately measuring the contribution 
# from selection mismatch (magnitude of false negatives + false positives)
# and estimatione rror (magnitude of error in correctly selected for coefficients)
def stratified_estimation_error(beta, beta_hat, threshold = False):
    beta, beta_hat = tile_beta(beta, beta_hat)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    fn_ee = np.zeros(beta_hat.shape[0])
    fp_ee = np.zeros(beta_hat.shape[0])
    estimation_ee = np.zeros(beta_hat.shape[0])
    
    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()

        common_support = np.bitwise_and(b != 0, bhat != 0)
        
        zerob = bhat[(b == 0)].ravel()
        false_positives = zerob[np.nonzero(zerob)]
        
        zerobhat = b[(bhat == 0).ravel()]
        false_negatives = zerobhat[np.nonzero(zerobhat)]
        fn_ee[i] = np.sum(np.abs(false_negatives))
        fp_ee[i] = np.sum(np.abs(false_positives))
        p = bhat[common_support].size
        if p > 0:
            estimation_ee[i] = np.sqrt(np.sum(np.power(b[common_support] - \
                                  bhat[common_support], 2)))
        else:
            estimation_ee[i] = 0

    return fn_ee, fp_ee, estimation_ee


def tile_beta(beta, beta_hat):

    if np.ndim(beta_hat) == 1:
        beta_hat = beta_hat[np.newaxis, :]

    if np.ndim(beta) == 1:
        beta = beta[np.newaxis, :]

    if beta.shape != beta_hat.shape: 
        beta = np.tile(beta, [int(beta_hat.shape[0]/beta.shape[0]), 1])

    return beta, beta_hat
