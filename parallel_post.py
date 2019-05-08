import numpy as np
import h5py
from scipy.stats import pearsonr
from mpi4py import MPI
import itertools
import time
import pdb
from pyuoi.mpi_utils import Gatherv_rows
# The index i is removed from np.arange(length). The index j then 
# corresponds to a different index in the 
def map_idxs(i, j):
    if j > i:
        idx = j + 1
    else:
        idx = j
    return idx

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.rank
    numproc = comm.Get_size()    

    f1 = h5py.File('CV_Lasso_a1.h5', 'r')
    f2 = h5py.File('EN_a1.h5', 'r')
    f3 = h5py.File('UoILasso_a1.h5', 'r')

    iter_idx = list(itertools.product(np.arange(5), np.arange(10), np.arange(127)))

    # Chunk
    chunk_list = np.array_split(iter_idx, numproc)
    chunk_idx = rank 
    num_tasks = len(chunk_list[chunk_idx])

    r1 = np.zeros(num_tasks)
    r2 = np.zeros(num_tasks)
    r3 = np.zeros(num_tasks)

    for i in range(num_tasks):
        idx_tuple = chunk_list[chunk_idx][i]
        c1 = []
        db1 = []

        c2 = []
        db2 = []

        c3 = []
        db3 = []
        t0 = time.time()
        for j in np.arange(127):
            if idx_tuple[2] == j:
                continue
            idx = map_idxs(idx_tuple[1], j)
            c1.append(f1['sigma_hats'][0, idx_tuple[2], idx])
            db1.append(np.abs(f1['coefs'][0, idx_tuple[0], idx_tuple[1], idx_tuple[2]] -\
                              f1['coefs'][0, idx_tuple[0], idx_tuple[1], j]))
            c2.append(f2['sigma_hats'][0, idx_tuple[2], idx])
            db2.append(np.abs(f2['coefs'][0, idx_tuple[0], idx_tuple[1], idx_tuple[2]] -\
                              f2['coefs'][0, idx_tuple[0], idx_tuple[1], j]))
            c3.append(f3['sigma_hats'][0, idx_tuple[2], idx])
            db3.append(np.abs(f3['coefs'][0, idx_tuple[0], idx_tuple[1], idx_tuple[2]] -\
                              f3['coefs'][0, idx_tuple[0], idx_tuple[1], j]))

        r1[i] = pearsonr(c1, db1)[0]
        r2[i] = pearsonr(c2, db2)[0]
        r3[i] = pearsonr(c3, db3)[0]
        print('%f s; %d/%d' % (time.time() - t0, i, num_tasks))
    # Gather 
    r1 = Gatherv_rows(r1, comm, root = 0)
    r2 = Gatherv_rows(r2, comm, root = 0)
    r3 = Gatherv_rows(r3, comm, root = 0)
    if rank == 0:
        with h5py.File('parallel_pro.h5', 'w') as f:

            f['r1'] = r1
            f['r2'] = r2
            f['r3'] = r3
