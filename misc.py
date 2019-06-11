import numpy as np
from scipy.optimize import broyden1
import itertools
import pdb
from scipy.optimize import minimize
import pickle
import struct
import networkx as nx
from networkx.algorithms.cluster import average_clustering
import utils
    

# For a list of dictionaries, group them by sets in which the entries differ by only the 
# value of the given key. If key is not contained in the dictionaries, the behavior of
# this function is to return the unique list of dictionaries and the indices that those values
# are found in the original list
def group_dictionaries(dicts, key):

    # Collect all the unique combinations of values, excluding the value of key
    groups = []
    group_idxs = []

    for i, dict_ in enumerate(dicts):
        values = []
        for k, v in dict_.items():
            if k == key:
                continue
            values.append(v)
        
        # Does this combination of values exist in groups?
        if values in groups:
            group_idxs[groups.index(values)].append(i)
        else:
            groups.append(values)
            group_idxs.append([i])

        pdb.set_trace()

    return groups, group_idxs

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

    y = np.reshape(y, (n_samples, 1))

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
def get_cov_list(n_features, n, correlation, block_size, L, n_supplement = 0):

    block_comb = list(itertools.product(correlation, block_size))
 
    avg_block_cov = np.zeros(len(block_comb))
    avg_exp_cov = np.zeros(len(L))
    for i, comb in enumerate(block_comb):
        avg_block_cov[i] = calc_avg_cov(n_features, comb[0], 
                                        comb[1], 1, 0)
    
    for i, l in enumerate(L):
        avg_exp_cov[i] = calc_avg_cov(n_features, 1, n_features, l, 1)

    avg_cov = np.concatenate([avg_block_cov, avg_exp_cov])

    # Take the range of available correlations and divide it into 
    # n equally spaced samples
    cmax = np.max(avg_cov)
    cmin = np.min(avg_cov[avg_cov > 0])

    # Log sample the number line
    coords = np.linspace(cmin, 0.3, 100 * n)

    final_comb = []
    final_comb.extend([(c[0], c[1], 1, 0) for c in block_comb])
    final_comb.extend([(1, n_features, l, 1) for l in L])
    
    starting_length = len(final_comb)

    # Keep track of the usage of the various covariance matrices
    possible_combs = list(itertools.product(block_comb, L))
    comb_counts = np.zeros(len(possible_combs))

    # Add an average covariance to the point that is farthest 
    # away from any neighbors on that iteration
    for i in range(n - starting_length):

        target_cov = max_separation(coords, avg_cov)
        avg_cov = np.append(avg_cov, target_cov)
                
        # Return the index of which covariance matrices to interpolate between
        comb_idx = get_best_comb(target_cov, avg_block_cov, avg_exp_cov, 
                                            comb_counts)
        comb_counts[comb_idx] += 1
        block_cov = avg_block_cov[block_comb.index(possible_combs[comb_idx][0])]
        exp_cov = avg_exp_cov[L.index(possible_combs[comb_idx][1])]
        
        t = (target_cov - block_cov)/(exp_cov - block_cov)
        final_comb.append(possible_combs[comb_idx][0] + (possible_combs[comb_idx][1], t))

        
    # Continue generating interpolations until all combinations have participated in one (potentially expensive):
#     while np.any(comb_counts == 0):
        
#         target_cov = max_separation(coords, avg_cov)
#         avg_cov = np.append(avg_cov, target_cov)
                
#         # Return the index of which covariance matrices to interpolate between
#         comb_idx = get_best_comb(target_cov, avg_block_cov, avg_exp_cov, 
#                                             comb_counts)
#         comb_counts[comb_idx] += 1
#         block_cov = avg_block_cov[block_comb.index(possible_combs[comb_idx][0])]
#         exp_cov = avg_exp_cov[L.index(possible_combs[comb_idx][1])]
        
#         t = (target_cov - block_cov)/(exp_cov - block_cov)
#         final_comb.append(possible_combs[comb_idx][0] + (possible_combs[comb_idx][1], t))
#         print(len(final_comb))
        
    # The strategy employed above is not guaranteed to let all combination participate. Therefore, if 
    # n_supplement > 0, generate an additional number of covariance matrices to even out the numbers.
    if n_supplement > 0:

        for i in range(n_supplement):            
            target_cov = max_separation(coords, avg_cov)
            
            comb_idx = np.argmin(comb_counts)
            comb_counts[comb_idx] += 1
            block_cov = avg_block_cov[block_comb.index(possible_combs[comb_idx][0])]
            exp_cov = avg_exp_cov[L.index(possible_combs[comb_idx][1])]
            
            # Bias the interpolation to be towards the direction of block_cov as this is what
            # is generally underpresented
            t = np.random.uniform(0.1, 0.5)
            np.append(avg_cov, block_cov * (1 - t) + exp_cov * t)
            final_comb.append(possible_combs[comb_idx][0] + (possible_combs[comb_idx][1], t))
    return final_comb, np.sort(avg_cov)    

# Return the element from x that is most separated from 
# its nearest neighbor in y, as well as that element's
# nearest neighbors in y1 and y2, which y is the union of
def max_separation(x, y):
    distances = np.zeros(x.size)
    for ix, xx in enumerate(x):
        distances[ix] = np.min(np.abs(y - xx))
    
    xsep = x[np.argmax(distances)] 

    return xsep

# First assemble all viable combinations, then return that which has been 
# featured the least so far
def get_best_comb(target, avg_block, avg_exp, comb_counts):
    
    comb_viable = np.zeros(len(comb_counts))
    
    for i, comb in enumerate(list(itertools.product(np.arange(avg_block.size), np.arange(avg_exp.size)))):
        t = (target - avg_block[comb[0]])/(avg_exp[comb[1]] - avg_block[comb[0]])
        if t > 0 and t < 1:
            comb_viable[i] = 1    
    
    comb_viable = comb_viable != 0
                    
    # Here parent_idx gives the index of lowest comb_count subject to the constraint that that 
    # particular combination is viable
    subset_idx = np.argmin(comb_counts[comb_viable])
    parent_idx = np.arange(comb_counts.size)[comb_viable][subset_idx]
        
    return parent_idx

# Utility function to shorten number of reps in a param file to facilitate testing
def shorten_param_file(param_path, target_path, n):
    f = open(param_path, 'rb')
    n_tasks = pickle.load(f)
    n_features = pickle.load(f)
    params = []
    
    n_tasks = min(n, n_tasks)
    for i in range(n_tasks):
        params.append(pickle.load(f))
    
    f.close()
    
    f = open(target_path, 'wb')
    f.write(pickle.dumps(n_tasks))
    f.write(pickle.dumps(n_features))

    for param in params:
        f.write(pickle.dumps(param))
    
    f.close()

# Return the dictionary at index in our indexed pickle files
def unpack_pickle(f, index):
    f.seek(0, 0)
    index_loc = f.read(8)
    index_loc = struct.unpack('L', index_loc)[0]
    f.seek(index_loc, 0)
    index_list = pickle.load(f)
    f.seek(index_list[index], 0)
    params = pickle.load(f)
    f.seek(0, 0)
    return params

def calc_clustering(n_features, correlation, block_size, L, t):
    # Gerneate the covariance matrix
    cov = utils.gen_covariance(n_features, correlation, block_size, L, t)
    # Convert to networkx graph
    G = nx.from_numpy_array(cov)
    avg_clustering = average_clustering(G, weight = 'weight')
    return avg_clustering
