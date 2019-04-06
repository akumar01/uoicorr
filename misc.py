import numpy as np
from scipy.optimize import root_scalar, broyden1

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
