from argparse import ArgumentParser
from utils import gen_covariance
import networkx as nx
import pickle
from networkx.algorithms.cluster import clustering
from networkx.algorithms.assortativity.correlation import degree_pearson_correlation_coefficient

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
print('calculating unweighted clustering')
unweighted_clustering = clustering(G)

# Calculate the weighted clustering
print('calculating weighted clustering')
weighted_clustering = clustering(G, weight = 'weight')

# Calculate the unweighted assortativity
print('calculating unweighted assortativity')
unweighted_assortativity = degree_pearson_correlation_coefficient(G)

# Weighted assortativity
print('calculating weighted assortativity')
weighted_assortativity = degree_pearson_correlation_coefficient(G, weight = 'weight')

# Save away
with open(args.results_file, 'wb') as f:
	f.write(pickle.dumps({'correlation': args.correlation, 
					   'block_size': args.block_size,
					   'L' : args.L, 
					   't' : args.t}))
	f.write(pickle.dumps(unweighted_clustering))
	f.write(pickle.dumps(weighted_clustering))
	f.write(pickle.dumps(unweighted_assortativity))
	f.write(pickle.dumps(weighted_assortativity))
 
