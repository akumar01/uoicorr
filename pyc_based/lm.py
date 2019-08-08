import pycasso
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model.coordinate_descent import _alpha_grid

# Pycasso solver wrapper with minimal class structure to interface with UoI
class PycassoLasso():

    def __init__(self, alphas=None, fit_intercept = False, max_iter = 1000):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.alphas = alphas
    def init_solver(self, X, y, alphas):

        self.solver = pycasso.Solver(X, y, family = 'gaussian', 
                      useintercept = self.fit_intercept, lambdas = alphas,
                      penalty = 'l1', max_ite = self.max_iter)

    def fit(self, X, y):

        if self.alphas is None:
            print('Set alphas before fitting!')
            return
            
        self.init_solver(X, y, self.alphas)
        self.solver.train()
        # Coefs across the entire solution path
        self.coef_ = self.solver.result['beta']

# Pycasso solver wrapper with minimal class structure to interface with UoI
class PycassoElasticNet():

    def __init__(self, fit_intercept = False, max_iter = 1000,
                 lambda1 = None, lambda2 = None):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def init_reg_params(self, X, y):

        # Set lambda2 using a RidgeCV fit
        if self.lambda2 is None:
            rdge = RidgeCV(alphas = np.linspace(1e-5, 100, 500)).fit(X, y)
            self.lambda2 = rdge.alpha_

        self.dummy_path = False

        if self.lambda1 is None:
            self.lambda1 = _alpha_grid(X, y, n_alphas = 100)
        else:
            if np.isscalar(self.lambda1):
                self.lambda1 = np.array([self.lambda1])
            lambda1 = np.flipud(np.sort(self.lambda1))
            if lambda1.size < 3:
                lambda1 = np.sort(lambda1)
                # Create a dummy path for the path solver
                self.dummy_path = True
                self.pathlength = lambda1.size
                while lambda1.size < 3:
                    lambda1 = np.append(lambda1, lambda1[-1]/2)
            self.lambda1 = lambda1

    def init_solver(self, X, y, lambda1 = None, lambda2 = None):

        self.init_reg_params(X, y)

        # We solve for an entire elastic net path with a fixed lambda2
        # For the given fixed lambda1, we modify the dataset to allow 
        # for the solution of a lasso-like problem
        xx, yy = augment_data(X, y, self.lambda2)

        # Augmented regularization parameters
        gamma = self.lambda1/np.sqrt(1 + self.lambda2)
        self.solver = pycasso.Solver(xx, yy, family = 'gaussian', 
                      useintercept = self.fit_intercept, lambdas = gamma,
                      penalty = 'l1', max_ite = self.max_iter)

    def fit(self, X, y):
        self.init_solver(X, y, self.lambda1, self.lambda2)
        self.solver.train()
        # Coefs across the entire solution path
        beta_naive = self.solver.result['beta']
    
        if self.dummy_path:
            beta_naive = beta_naive[:self.pathlength, :]

        # Rescale coefficients (eq. 11 of Elastic Net paper)
        self.coef_ = np.sqrt(1 + self.lambda2) * beta_naive

        # Record regularization parameters
        reg_params = np.zeros((self.lambda1.size, 2))
        reg_params[:, 0] = self.lambda2
        reg_params[:, 1] = self.lambda1

        self.reg_params = reg_params

# Augment data so ElasticNet becomes an l1 regularization problem 
def augment_data(X, y, l2):

    n_samples, n_features = X.shape

    if y.ndim == 1:
        y = y[:, np.newaxis]

    # Augment the data
    XX = 1/np.sqrt(1 + l2) * np.vstack([X, np.sqrt(2 * n_samples) * np.sqrt(l2) * np.eye(n_features)])
    yy = np.vstack([y, np.zeros((n_features, 1))])
 
    return XX, yy