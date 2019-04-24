import numpy as np
from cov_estimators import banding, inverse_banding, thresholding
import itertools
from mpi4py import MPI
from pyuoi.mpi_utils import Gatherv_rows
import pickle
from utils import gen_data, interpolate_covariance, gen_beta2
from sklearn.covariance import EmpiricalCovariance, GraphicalLassoCV, LedoitWolf, MinCovDet, OAS, ShrunkCovariance
import pdb

#n_samples = np.array([12, 20, 30])
n_samples= np.array([50, 100, 200])
n_features = 100
block_size = np.array([5, 10])
#block_size = np.array([25])
block_correlation = np.array([0.05, 0.1, 0.25, 0.5])
sparsity = 1
L = np.array([5, 25])
betawidth = 5

shape = (n_samples.size, block_size.size, block_correlation.size, L.size, 3)
methods = ['empirical', 'ShrunkCovariance', 'LedoitWolf', 'OAS', 'banding', 'inverse_banding', 'thresholding']

# Estimators
empcov = EmpiricalCovariance(assume_centered = True)
glcv = GraphicalLassoCV(cv = 5, assume_centered = True)
shrunk = ShrunkCovariance(assume_centered = True)
lw = LedoitWolf(assume_centered = True)
oas = OAS(assume_centered = True)

iter_idx = list(itertools.product(*[np.arange(length) for length in shape[:-1]]))

# Paralllelize
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()    

# Chunk:
chunk_list = np.array_split(iter_idx, numproc)
num_tasks = len(chunk_list[rank])
errors = np.zeros((num_tasks, 3, len(methods)))
spectrum_errors = np.zeros((num_tasks, 3, len(methods)))

print('rank: %d' % rank)

for ii, idx in enumerate(chunk_list[rank]):
    nf = n_features
    s = sparsity
    bs = block_size[idx[1]] 
    ns = n_samples[idx[0]]
    bc = block_correlation[idx[2]]
    l = L[idx[3]]

    beta = gen_beta2(nf, bs, s, betawidth)
    sigmas = interpolate_covariance('block_covariance', 'exp_falloff', interp_coeffs = np.array([0, 0.5, 1]),
             n_features = nf, cov_type1_args = {'block_size': bs, 'correlation': bc},
             cov_type2_args = {'L' : l})

    for i, sigma in enumerate(sigmas):
        X, X_test, y, y_test = gen_data(n_samples = ns, n_features = nf,
                                       covariance = sigma['sigma'], beta = beta)
        print('Process %d began inner iteration %d' % (rank, i))
#    try:
        empcov.fit(X)
        errors[ii, i, 0] = empcov.error_norm(sigma['sigma'])
        spectrum_errors[ii, i, 0] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(empcov.covariance_))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
#     except:
#         errors[ii, i, 0] = np.nan
#         spectrum_errors[ii, i, 0] = np.nan
        print('Process %d did empirical covariance'% rank)

#    try:
        shrunk.fit(X)
        errors[ii, i, 1] = shrunk.error_norm(sigma['sigma'])
        spectrum_errors[ii, i, 1] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(shrunk.covariance_))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
        print('Process %d did shrunk covariance' % rank)

#     except:
#         errors[ii, i, 1] = np.nan
#         spectrum_errors[ii, i, 1] = np.nan
#    try:
        lw.fit(X)
        errors[ii, i, 2] = lw.error_norm(sigma['sigma'])
        spectrum_errors[ii, i, 2] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(lw.covariance_))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
#     except:
#         errors[ii, i, 2] = np.nan
#         spectrum_errors[ii, i, 2] = np.nan
#    try:
        oas.fit(X)
        errors[ii, i, 3] = oas.error_norm(sigma['sigma'])
        spectrum_errors[ii, i, 3] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(oas.covariance_))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
#     except:
#         errors[ii, i, 3] = np.nan
#         spectrum_errors[ii, i, 3] = np.nan
#    try:
        print('Process %d did OAS covariance' % rank)

        sigma_hat = banding(X)
        errors[ii, i, 4] = np.linalg.norm(sigma['sigma'] - sigma_hat)
        spectrum_errors[ii, i, 4] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(sigma_hat))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
        print('Process %d did banding covariance' % rank)
        #     except:
#         pdb.set_trace()
#         errors[ii, i, 4] = np.nan
#         spectrum_errors[ii, i, 4] = np.nan
#    try:
        # try:
        #     sigma_hat = inverse_banding(X)
        #     errors[ii, i, 5] = np.linalg.norm(sigma['sigma'] - sigma_hat)
        #     spectrum_errors[ii, i, 5] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(sigma_hat))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
        #     print('Process %d did inv banding covariance' % rank)
        # except:
        #     errors[ii, i, 5] = np.nan
        #     spectrum_errors[ii, i, 5] = np.nan
        #     except:
#         pdb.set_trace()
#         errors[ii, i, 5] = np.nan
#         spectrum_errors[ii, i, 5] = np.nan
#    try:
        try:
            sigma_hat = thresholding(X)
            errors[ii, i, 6] = np.linalg.norm(sigma['sigma'] - sigma_hat)
            spectrum_errors[ii, i, 6] = np.linalg.norm(np.sort(np.real(np.linalg.eigvals(sigma_hat))) - np.sort(np.real(np.linalg.eigvals(sigma['sigma']))))
            print('Process %d did thresholding covariance' % rank)
        except:
            errors[ii, i, 6] = np.nan
            spectrum_errors[ii, i, 6] = np.nan
#     except:
#         pdb.set_trace()
#         errors[ii, i, 6] = np.nan
#         spectrum_errors[ii, i, 6] = np.nan
            
            
    print('Process %d has completed task %d/%d' % (rank, ii + 1, num_tasks))

# Gather together 
errors = errors[np.newaxis, :]
spectrum_errors = spectrum_errors[np.newaxis, :]

errors = Gatherv_rows(errors, comm, root = 0)
spectrum_errors = Gatherv_rows(spectrum_errors, comm, root = 0)

if rank == 0:
    # Create new arrays that align with the expected shape
    try:
        errors_full = np.zeros(shape + (len(methods),))
        spectrum_errors_full = np.zeros(shape + (len(methods),))

        for i in range(len(chunk_list)):
            for j in range(len(chunk_list[i])):
                errors_full[chunk_list[i][j][0], chunk_list[i][j][1], chunk_list[i][j][2], chunk_list[i][j][3], :] = errors[i, j, :]
                spectrum_errors_full[chunk_list[i][j][0], chunk_list[i][j][1], chunk_list[i][j][2], chunk_list[i][j][3], :] = spectrum_errors[i, j, :]

        with open('all_estimators4.dat', 'wb') as f:
            pickle.dump(errors_full, f)
            pickle.dump(spectrum_errors_full, f)
            pickle.dump(errors, f)
            pickle.dump(spectrum_errors, f)
            pickle.dump(chunk_list, f)
    except:
       
        with open('all_estimators_backup4.dat', 'wb') as f:
            pickle.dump(errors, f)
            pickle.dump(spectrum_errors, f)
            pickle.dump(chunk_list, f)
