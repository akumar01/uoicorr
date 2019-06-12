import numpy as np
from mpi4py import MPI
from pyuoi.linear_model import UoI_Lasso
from utils import gen_covariance, gen_beta2, gen_data, selection_accuracy
import time
import pdb

comm = MPI.COMM_WORLD
root = 0
rank = comm.rank
size = comm.size

# Time UoI Lasso with LARS for increasing p. Keep sparsity = 1 and introduce correlations to # test the worst-case runtimes 

# Keep track of selection accuracy and r2 to make sure there is no drop-off in performance

n_features = np.array([100, 200, 300, 400, 500, 1000, 2000])
n_samples = 5 * n_features

r2 = np.zeros(n_features.size)
sa = np.zeros(n_features.size)
    
for i, nf in enumerate(n_features):
    
    sigma = gen_covariance(nf, 0, nf, 10, 0.25)
    beta = gen_beta2(n_features = nf, block_size = nf, sparsity = 0.2)
    X, X_test, y, y_test, _ = gen_data(n_samples[i], nf, covariance = sigma, beta = beta)
    
    uoil = UoI_Lasso(comm = comm, stability_selection = 0.75)
    t0 = time.time()
    uoil.fit(X, y.ravel())

    if rank == 0:
        pdb.set_trace()
        print('n_features: %d' % nf)
        print('Time: %f' % (time.time() - t0))    
        r2[i] = uoil.score(X_test, y_test)
        sa[i] = selection_accuracy(beta.ravel(), uoil.coef_.ravel())
        print('r2 score: %f' % r2[i])
        print('selection accuracy: %f' % sa[i])
