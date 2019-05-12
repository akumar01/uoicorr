from argparse import ArgumentParser
from utils import gen_covariance
import networkx as nx
import h5py
from networkx.algorithms.cluster import clustering
from networkx.algorithms.assortativity.correlation import degree_correlation_coefficient

# Parse command line arguments
p = ArgumentParser()
p.add_argument('results_file')
p.add_argument('-c', '--correlation', type = float)
p.add_argument('-b', '--block_size', type = int)
p.add_argument('-L', type = float)
p.add_argument('-t', type = float)
args = p.parse_args()

# Generate the covariance matrix
cov = gen_covariance(1000, args.correlation, args.block_size, args.L, args.t)

# Convert to graph
G = nx.from_numpy_matrix(cov)

# Calculate the unweighted clustering: 
unweighted_clustering = clustering(G)

# Calculate the weighted clustering
weighted_clustering = clustering(G, weight = 'weight')

# Calculate the unweighted assortativity
unweighted_assortativity = degree_correlation_coefficient(G)

# Weighted assortativity
weighted_assortativity = degree_correlation_coefficient(G, weight = 'weight')

# Save away
with h5py.File(results_file, 'w') as f:
	f['cov_params'] = {'correlation': args.correlation, 
					   'block_size': args.block_size,
					   'L' : args.L, 
					   't' : args.t}
	f['unweighted_clustering'] = unweighted_clustering
	f['weighted_clustering'] = weighted_clustering
	f['unweighted_assortativity'] = unweighted_assortativtiy
	f['weighted_assortativity'] = weighted_assortativity
 
