# Covariance Estimators
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

# Set all elements a distance > k from the diagonal to zero
def banded_matrix(M, k):
    mask = np.invert((np.tri(M.shape[0], k = - k - 1) + np.tri(M.shape[0], k = -k - 1).T) != 0)
    M[mask] = 0
    return M

# Simply cutoff elements a certain distance from the diagonal. 
# Choose this distance by estimation of the risk through re-sampling
def banding(X, n_splits):
    n = X.shape[0]
    p = X.shape[1]

    # Mean center X: 
    X = X - np.mean(X, axis = 0)

    # Possible values of k:
    K = np.arange(0, p)
    risk = np.zeros(p)


    for k in K:
        for i in range(n_splits):
            X1, X2 = train_test_split(X, train_size = 0.33)
            banded_cov = banded_matrix(1/X1.shape[0] * X1.T @ X1)
            emprirical_cov = 1/X2.shape[0] * X2.T @ X2
            risk[k] += 1/n_splits * np.sum(np.abs(banded_cov - emprirical_cov))

    # Choose k that minimizes the risk:
    k_f = K[np.argmin(risk)]
    return banded_matrix(1/n * X.T @ X, k_f)


def inverse_banding(X):
    n = X.shape[0]
    p = X.shape[1]

    # Mean center X: 
    X = X - np.mean(X, axis = 0)

    # Possible values of k:
    K = np.arange(0, min(n, p))
    risk = np.zeros(min(n, p))

    for k in K:
        for i in range(n_splits):
            X1, X2 = train_test_split(X, train_size = 0.33)
            empirical1 =1/X1.shape[0] * X1.T @ X1
            empirical2 = 1/X2.shape[0] * X2.T @ X2
            banded_inverse = banded_matrix(np.linagl.inv(empirical1), k)
            risk[k] += 1/n_splits * np.sum(np.abs(banded_inverse - np.linalg.inv(empirical2)))

    # Choose k that minimizes the risk:
    k_f = K[np.argmin(risk)]
    SigmaInv = banded_matrix(np.linalg.inv(1/n * X.T @ X), k_f)

    return np.linalg.inv(SigmaInv)


# First identify factor model by fitting latent factor model to data, 
# using cross validation to choose the factor size
def factor_model(X):
    n = X.shape[0]
    p = X.shape[1]
    n_factors = np.arange(1, p)
    model_scores = np.zeros(n_factors.size)
    fa = FactorAnalysis()
    for i, n in enumerate(n_factors):
        fa.n_components = n
        fa_scores[i] = np.mean(cross_val_score(fa, X, cv = 5))

    cv_nfactors = n_factors[np.argmax(fa_scores)]

    # Return the covariance estimate of a factor model


def UoIGraphicalLasso(X, y):
