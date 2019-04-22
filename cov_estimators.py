# Covariance Estimators
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from pyuoi.linear_model.base import AbstractUoILinearModel

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
    fa.n_components(cv_nfactors)
    fa.fit(X)
    return fa.get_covariance()

# Use UoI Lasso in place of Lasso in the Graphical Lasso algorithm
class UoIGraphicalLasso(AbstractUoILinearModel):

        def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, n_lambdas=48, stability_selection=1.,
                 eps=1e-3, warm_start=True, estimation_score='frobenius',
                 random_state=None, max_iter=1000,
                 comm=None):
            super(UoIGraphicalLasso, self).__init__(
                n_boots_sel = n_boots_sel, n_boots_est = n_boots_est,
                selection_frac = selection_frac, estimation_frac = estimation_frac,
                stability_selection = stability_selection, random_state = random_state,
                comm = comm
                )
            self.n_lambdas = n_lambdas
            self.eps = eps
            self.__selection_lm = Lasso()