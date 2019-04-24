import numpy as np
from scipy.optimize import broyden1
import itertools
import pdb

# Solve for the L needed to yield a desired average correlation
# for exponential falloff design
def solve_L(p, avg_cov):

    f = lambda L : np.array(1/p**2 * (p + 2 * (np.exp(1/L[0]) * (p + np.exp(-p/L[0]) -1 ) - p)\
                                                    /(np.exp(1/L[0]) - 1)**2) - avg_cov)

#    Lsol = root_scalar(f, method='brentq', bracket = [1e-5, 10 * p])
    Lsol = broyden1(f, [1.0])
    return Lsol[0]

# Solve for the interpolation strength needed to yield a desired average correlation
# for interpolated design
def solve_t(avg_cov, sigma1, sigma2):

    t = (avg_cov - np.mean(sigma1))/(np.mean(sigma2) - np.mean(sigma1))
    if t > 1 or t < 0:
        raise Exception('Cannot interpolate between provided matrices to yield desired\
                        average correlation')    
    return t

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

# Calculate the average covariance given the 5 parameters that define 
# an interpolated covariance matrix
# Trim off the diagonal component
def calc_avg_cov(p, correlation, block_size, L, t):

    # Average correlation of th
    c1 = correlation * (block_size - 1)/p
    c2 = 1/p**2 * (2 * (np.exp(1/L) * (p + np.exp(-p/L) -1 ) - p)\
                                                    /(np.exp(1/L) - 1)**2)

    return (1- t) * c1 + t * c2

# Return n total covariance matrices that uniformly sample the
# average correlation space, 
def get_cov_list(n_features, n, correlation, block_size, L):

    block_comb = list(itertools.product(correlation, block_size))
 
    avg_block_cov = np.zeros(len(block_comb))
    avg_exp_cov = np.zeros(len(L))

    for i, comb in enumerate(block_comb):
        avg_block_cov[i] = calc_avg_cov(n_features, comb[0], 
                                        comb[1], 1, 0)

    for i, l in enumerate(L):
        avg_exp_cov[i] = calc_avg_cov(n_features, 1, 1, l, 1)

    avg_cov = np.concatenate([avg_block_cov, avg_exp_cov])

    # Take the range of available correlations and divide it into 
    # n equally spaced samples
    cmax = np.max(avg_cov)
    cmin = np.min(avg_cov[avg_cov > 0])

    # Log sample the number line
    coords = np.linspace(0.1, 0.3, 100 * n)

    final_comb = []
    final_comb.extend([(c[0], c[1], 1, 0) for c in block_comb])
    final_comb.extend([(1, 1, l, 1) for l in L])

    starting_length = len(final_comb)

    # Add an average covariance to the point that is farthest 
    # away from any neighbors on that iteration
    for i in range(n - starting_length):

        target_cov = max_separation(coords, avg_cov)
        avg_cov = np.append(avg_cov, target_cov)
        # Grab the parameters of the corresponding covariance matrices
        idxblock = np.random.choice(np.arange(len(avg_block_cov)))
        idxexp = np.random.choice(np.arange(len(avg_exp_cov)))

        t = (target_cov - avg_block_cov[idxblock])/(avg_exp_cov[idxexp] - avg_block_cov[idxblock])

        while t > 1 or t < 0 or np.isnan(t):
            # Grab the parameters of the corresponding covariance matrices
            idxblock = np.random.choice(np.arange(len(avg_block_cov)))
            idxexp = np.random.choice(np.arange(len(avg_exp_cov)))
            t = (target_cov - avg_block_cov[idxblock])/(avg_exp_cov[idxexp] - avg_block_cov[idxblock])

        final_comb.append((block_comb[idxblock]) + (L[idxexp], t))

    return final_comb    

# Return the element from x that is most separated from 
# its nearest neighbor in y, as well as that element's
# nearest neighbors in y1 and y2, which y is the union of
def max_separation(x, y):
    distances = np.zeros(x.size)
    for ix, xx in enumerate(x):
        distances[ix] = np.min(np.abs(y - xx))
    
    xsep = x[np.argmax(distances)] 

    return xsep

# Conditional max:
def cmax(x):
    if not list(x):
        return np.nan
    else:
        return np.max(x)

def cmin(x):
    if not list(x):
        return np.nan
    else:
        return np.min(x)


