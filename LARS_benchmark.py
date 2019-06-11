import numpy as np
from mpi4py import MPI
from pyuoi.linear_model import UoI_Lasso
from utils import gen_covariance, gen_beta2, gen_data, selection_accuracy
from sklearn.metrics import r2_score

comm = MPI.COMM_WORLD
root = 0
rank = comm.rank
size = comm.size

# Time UoI Lasso with LARS for increasing p. Keep sparsity = 1 and introduce correlations to # test the worst-case runtimes 

# Keep track of selection accuracy and r2 to make sure there is no drop-off in performance

n_features = np.array([100, 200, 300, 400, 500, 1000, 2000, 5000])
n_samples = 5 * n_features

r2 = np.zeros(n_features.size)
sa = np.zeros(n_features.size)
    
for i, nf in enumerate(n_features):
    
    sigma = gen_covariance(nf, 0, nf, 10, 0.25)
    beta = gen_beta2(n_features = nf, block_size = nf, sparsity = 1)
    X, X_test, y, y_test, _ = gen_data(n_samples[i], nf, covariance = sigma, beta = beta)
    
    uoil = UoI_Lasso()
    print('n_features: %d' % nf)
    uoil.fit(X, y.ravel())
    
    r2_score[i] = uoil.score(y_test, X_test)
    sa[i] = selection_accuracy(beta.ravel(), uoil.coef_.ravel())

