import sys, os
from mpi4py import MPI
import numpy as np
from pyuoi.mpi_utils import Gatherv_rows, Bcast_from_root
from utils import gen_beta2, gen_data
from sklearn.metrics import r2_score

# Initialize comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()

ranks = np.arange(numproc)
split_ranks = np.array_split(ranks, 20)
color = [i for i in np.arange(20) if rank in split_ranks[i]][0]
subcomm_roots = [split_ranks[i][0] for i in np.arange(20)]

subcomm = comm.Split(color, rank)

rank = color
nchunks = 20
subrank = subcomm.rank
numproc = subcomm.Get_size()

# Create a group including the root of each subcomm. 
global_group = comm.Get_group()
root_group = MPI.Group.Incl(global_group, subcomm_roots)
roots_comm = comm.Create(root_group)

print('comm rank: %d, subcomm rank: %d, color: %d' % (comm.rank, subcomm.rank, color))

if subrank == 0: 
        r2_scores = np.zeros(5)
        print('roots_comm rank: %d' % roots_comm.rank)

for i in range(1):
    if subrank == 0:
        # Generate data
        beta = gen_beta2()
        X, X_test, y, y_test = gen_data(180, beta = beta)
    else:
        X = None
        X_test = None
        y = None 
        y_test = None
        
    X = Bcast_from_root(X, subcomm)
    X_test = Bcast_from_root(X_test, subcomm)
    y = Bcast_from_root(y, subcomm)
    y_test = Bcast_from_root(y_test, subcomm)

    if subrank == 0:
        r2_scores[i] = r2_score(y_test, np.dot(X_test, beta))
        if roots_comm.rank == 0:
     	    print(beta)
 	#print(r2_scores[i])
if subrank == 0:
        print('Starting gather')
        r2_scores = Gatherv_rows(r2_scores, roots_comm, root = 0)
        print (r2_scores)
        
