from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid
import numpy as np
import pycasso
from .base import AbstractUoILinearRegressor


class UoI_Lasso(AbstractUoILinearRegressor, LinearRegression):
    """ UoI Lasso model.

    Parameters
    ----------
    n_boots_sel : int, default 48
        The number of data bootstraps to use in the selection module.
        Increasing this number will make selection more strict.

    n_boots_est : int, default 48
        The number of data bootstraps to use in the estimation module.
        Increasing this number will relax selection and decrease variance.

    n_lambdas : int, default 48
        The number of regularization values to use for selection.

    alpha : list or ndarray of floats
        The parameter that trades off L1 versus L2 regularization for a given
        lambda.

    selection_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the selection module. Small values of this parameter
        imply larger "perturbations" to the dataset.

    estimation_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the estimation module. The remaining data is used
        to obtain validation scores. Small values of this parameters imply
        larger "perturbations" to the dataset.

    stability_selection : int, float, or array-like, default 1
        If int, treated as the number of bootstraps that a feature must
        appear in to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.

    estimation_score : str "r2" | "AIC", | "AICc" | "BIC"
        Objective used to choose the best estimates per bootstrap.

    warm_start : bool, default True
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution

    eps : float, default 1e-3
        Length of the lasso path. eps=1e-3 means that
        alpha_min / alpha_max = 1e-3

    copy_X : boolean, default True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_intercept : boolean, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    standardize : boolean, default False
        If True, the regressors X will be standardized before regression by
        subtracting the mean and dividing by their standard deviations.

    max_iter : int, default None
        Maximum number of iterations for iterative fitting methods.

    random_state : int, RandomState instance or None, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    comm : MPI communicator, default None
        If passed, the selection and estimation steps are parallelized.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    supports_ : array, shape
        boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, n_lambdas=48, stability_selection=1.,
                 estimation_score='r2', eps=1e-3, warm_start=True,
                 copy_X=True, fit_intercept=True, standardize=True,
                 max_iter=1000, random_state=None, comm=None, logger=None):
        super(UoI_Lasso, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            standardize=standardize,
            random_state=random_state,
            comm=comm,
            estimation_score=estimation_score,
            max_iter=max_iter,
            logger=logger
        )
        self.n_lambdas = n_lambdas
        self.eps = eps
        self._selection_lm = PycassoLasso(
            max_iter=max_iter,
            fit_intercept=fit_intercept)
        self._estimation_lm = LinearRegression(fit_intercept=fit_intercept)

    def get_reg_params(self, X, y):
        alphas = _alpha_grid(
            X=X, y=y,
            l1_ratio=1.0,
            fit_intercept=self.fit_intercept,
            eps=self.eps,
            n_alphas=self.n_lambdas)
        return [{'alpha': a} for a in alphas]

    # Overwrite base class selection sweep to accommodate Pycasso path-wise solution
    def uoi_selection_sweep(self, X, y, reg_param_values):

        n_param_values = len(reg_param_values)
        n_coef = self.get_n_coef(X, y)
        coefs = np.zeros((n_param_values, n_coef))
        alphas = np.array([reg_param_values[i]['alpha'] 
                           for i in range(len(reg_param_values))])

        self._selection_lm.fit(X, y, alphas)

        return self._selection_lm.coef_

# Pycasso solver wrapper with minimal class structure to interface with UoI
class PycassoLasso():

    def __init__(self, fit_intercept = False, max_iter = 1000):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def init_solver(self, X, y, alphas):

        self.solver = pycasso.Solver(X, y, family = 'gaussian', 
                      useintercept = self.fit_intercept, lambdas = alphas,
                      penalty = 'l1', max_ite = self.max_iter)

    def fit(self, X, y, alphas):

        self.init_solver(X, y, alphas)
        self.solver.train()
        # Coefs across the entire solution path
        self.coef_ = self.solver.result['beta']
