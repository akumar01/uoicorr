# Covariance Estimators
import numpy as np
import pdb
from sklearn.model_selection import KFold
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# from pyuoi.linear_model.base import AbstractUoILinearModel
from sklearn.covariance import GraphicalLasso, EmpiricalCovariance
import timeout_decorator

# Set all elements a distance > k from the diagonal to zero
def banded_matrix(M, k):
    mask = (np.tri(M.shape[0], k = - k - 1) + np.tri(M.shape[0], k = -k - 1).T) != 0
    M[mask] = 0
    return M

# Simply cutoff elements a certain distance from the diagonal. 
# Choose this distance by estimation of the risk through re-sampling
@timeout_decorator.timeout(60 * 60)
def banding(X, n_splits = 10, use_signals = False):
    n = X.shape[0]
    p = X.shape[1]

    # Mean center X: 
    X = X - np.mean(X, axis = 0)

    # Possible values of k:
    K = np.arange(0, p)
    risk = np.zeros(p)

    for k in K:
        for i in range(n_splits):
            X1, X2 = train_test_split(X, train_size = 0.33, test_size = 0.67)
            empirical_cov = EmpiricalCovariance().fit(X1).covariance_
            banded_cov = banded_matrix(empirical_cov, k)
            empirical_cov = EmpiricalCovariance().fit(X2).covariance_
            risk[k] += 1/n_splits * np.sum(np.abs(banded_cov - empirical_cov))

    # Choose k that minimizes the risk:
    k_f = K[np.argmin(risk)]
    empirical_full = EmpiricalCovariance().fit(X).covariance_
    return banded_matrix(empirical_full, k_f)

def modified_cholesky(X, k):
    Xhat = np.zeros((X.shape[0], X.shape[1]))
    A = np.zeros((X.shape[1], X.shape[1]))
    for j in range(X.shape[1]):
        if j == 0:
            continue
        Zj = X[:, max(0, j - k): j]
        # Caclulate the a_j coefficients:
        aj = np.linalg.pinv(Zj) @ X[:, j]  
        A[j, j - aj.size:j] = aj
        Xhat[:, j] = Zj @ aj
    epsilon = X - Xhat
    D = np.diag(np.var(epsilon, axis = 0))
    return (np.identity(X.shape[1]) - A).T @ np.linalg.inv(D) @\
           (np.identity(X.shape[1]) - A)

@timeout_decorator.timeout(60*60)
def inverse_banding(X, n_splits = 10, use_signals = False):
    n = X.shape[0]
    p = X.shape[1]

    # Mean center X: 
    X = X - np.mean(X, axis = 0)

    # Possible values of k:
    K = np.arange(1, min(n, p))
    risk = np.zeros(min(n, p) - 1)

    for i, k in enumerate(K):
        for i in range(n_splits):
            X1, X2 = train_test_split(X, train_size = 0.33, test_size = 0.67)
            sigma_k1 = modified_cholesky(X1, k)
            sigma_k2 = modified_cholesky(X2, X2.shape[1])
            risk[i] += 1/n_splits * np.sum(np.abs(sigma_k1 - sigma_k2))

    # Choose k that minimizes the risk:
    k_f = K[np.argmin(risk)]
    SigmaInv = modified_cholesky(X, k_f)
    return np.linalg.inv(SigmaInv)


# First identify factor model by fitting latent factor model to data, 
# using cross validation to choose the factor size
@timeout_decorator.timeout(60 * 60, use_signals = False)
def factor_model(X):
    n = X.shape[0]
    p = X.shape[1]
    n_factors = np.arange(1, p)
    model_scores = np.zeros(n_factors.size)
    fa = FactorAnalysis()
    fa_scores = np.zeros(n_factors.size)
    for i, n in enumerate(n_factors):
        fa.n_components = n
        fa_scores[i] = np.mean(cross_val_score(fa, X, cv = 5))

    cv_nfactors = n_factors[np.argmax(fa_scores)]
    fa.n_components = cv_nfactors
    fa.fit(X)
    return fa.get_covariance()

# Use UoI Lasso in place of Lasso in the Graphical Lasso algorithm
# class UoIGraphicalLasso(AbstractUoILinearModel):

#     # Rename covariance to coef_
#     class ModifiedGraphicalLasso(GraphicalLasso):
#         def __init__(self):
#             super(ModifiedGraphicalLasso, self).__init__()

#         def fit(X, y = None):
#             super(ModifiedGraphicalLasso, self).fit(X)
#             self.coef_ = self.covariance_.ravel()

#     class UnregGraphicalModel():
#         def __init__(self):
#             pass

#         def fit(X, y = None):
#             pass

#     def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
#              estimation_frac=0.9, n_lambdas=48, stability_selection=1.,
#              eps=1e-3, warm_start=True, random_state=None, max_iter=100,
#              comm=None):
#         super(UoIGraphicalLasso, self).__init__(
#             n_boots_sel = n_boots_sel, n_boots_est = n_boots_est,
#             selection_frac = selection_frac, estimation_frac = estimation_frac,
#             stability_selection = stability_selection, random_state = random_state,
#             comm = comm
#             )
#         self.n_lambdas = n_lambdas
#         self.eps = eps

#         self.__selection_lm = self.ModifiedGraphicalLasso(max_iter = max_iter)
#         self.__estimation_lm = self.UnregGraphicalModel()

#     # For selection
#     @property
#     def selection_lm(self):
#         return self.__selection_lm

#     @property
#     def estimation_lm(self):
#         return self._estimation_lm
    
#     def score_predictions():
#         pass

#     # Follow the same approach as sklearn's CVGraphicalLasso to select reg_parmas
#     def get_reg_params(self, X, y = None):
#         pass