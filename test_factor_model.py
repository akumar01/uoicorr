import numpy as np
from cov_estimators import factor_model
import itertools
from mpi4py import MPI
from pyuoi.mpi_utils import Gatherv_rows
import pickle
from utils import gen_data, interpolate_covariance, gen_beta2

n_samples = np.array([25, 50, 100, 150, 200, 400])
n_features = 200
block_size = np.array([10, 20, 40])
block_correlation = np.array([0.05, 0.1, 0.25, 0.5])
sparsity = 1
L = np.array([10])
betawidth = 5

shape = (n_samples.size, block_size.size, block_correlation.size, L.size, 2)

iter_idx = list(itertools.product(*[np.arange(length) for length in shape[:-1]]))

# Paralllelize
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()    

# Chunk:
chunk_list = np.array_split(iter_idx, numproc)
num_tasks = len(chunk_list[rank])
errors = np.zeros((num_tasks, 2))
spectrum_errors = np.zeros((num_tasks, 2))

print('rank: %d' % rank)

for ii, idx in enumerate(chunk_list[rank]):
    nf = n_features
    s = sparsity
    bs = block_size[idx[1]] 
    ns = n_samples[idx[0]]
    bc = block_correlation[idx[2]]
    l = L[idx[3]]

    beta = gen_beta2(nf, bs, s, betawidth)
    sigmas = interpolate_covariance('block_covariance', 'exp_falloff', interp_coeffs = np.array([0]),
             n_features = nf, cov_type1_args = {'block_size': bs, 'correlation': bc},
             cov_type2_args = {'L' : l})

    for i, sigma in enumerate(sigmas):
        X, X_test, y, y_test = gen_data(n_samples = ns, n_features = nf,
                                       covariance = sigma['sigma'], beta = beta)
        try:
            sigma_hat = factor_model(X)
            errors[ii, i] = np.linalg.norm(sigma['sigma'] - sigma_hat)
            spectrum_errors[ii, i] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(sigma_hat))) 
        								- np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
        except:
            print('Timeout!')
            errors[ii, i] = np.nan
            spectrum_errors[ii, i] = np.nan
    print('Process %d has completed task %d/%d' % (rank, ii + 1, num_tasks))
# Gather together 

errors = errors[np.newaxis, :]
spectrum_errors = spectrum_errors[np.newaxis, :]

errors = Gatherv_rows(errors, comm, root = 0)
spectrum_errors = Gatherv_rows(spectrum_errors, comm, root = 0)

if rank == 0:
    # Create new arrays that align with the expected shape
    errors_full = np.zeros(shape)
    spectrum_errors_full = np.zeros(shape)

    for i in range(len(chunk_list)):
        for j in range(len(chunk_list[i])):
            errors_full[chunk_list[i][j][0], chunk_list[i][j][1], chunk_list[i][j][2], chunk_list[i][j][3], :] = errors[i, j, :]
            spectrum_errors_full[chunk_list[i][j][0], chunk_list[i][j][1], chunk_list[i][j][2], chunk_list[i][j][3], :] = spectrum_errors[i, j, :]
    with open('factor_model_test.dat', 'wb') as f:
        pickle.dump(errors_full, f)
        pickle.dump(spectrum_errors_full, f)
        pickle.dump(errors, f)
        pickle.dump(spectrum_errors, f)
        pickle.dump(chunk_list, f)