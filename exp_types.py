import numpy as np
import pdb
import itertools
import time

from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import RidgeCV

from pyuoi.linear_model import UoI_Lasso
from pyuoi.linear_model import UoI_ElasticNet
from pyuoi.linear_model.mcp import UoI_MCP

from pyc_based.lm import PycassoLasso, PycassoElasticNet
from pyc_based.pycasso_cv import PycassoCV, PycassoGrid, PycEnCV

from r_based.slope import SLOPE as SLOPE_
from r_based.slope import SLOPE_CV

from selection import Selector, UoISelector

class CV_Lasso():

    @classmethod
    def run(self, X, y, args):

        n_alphas = args['n_alphas']
        # Run and store the entire solution path
        print('Fitting!')
        alphas = _alpha_grid(X, y.ravel(), n_alphas = n_alphas)
        lasso = PycassoLasso(alphas=alphas, fit_intercept=False)
        lasso.fit(X, y)
        results = {}
        results['coefs'] = lasso.coef_
        results['reg_params'] = alphas
        return results

class EN():

    @classmethod
    def run(self, X, y, args):

        alphas = _alpha_grid(X, y.ravel(), n_alphas=n_alphas)
        # Run RidgeCV to determine L2 penalty
        rdge = RidgeCV(alphas = np.linspace(1e-5, 100, 500)).fit(X, y)
        l2 = rdge.alpha_
        en = PycassoElasticNet(fit_intercept=False, lambda1=alphas, lambda2=l2)
        en.fit(X, y.ravel())
        results = {}
        results['coefs'] = en.coef_

        reg_params = np.zeros((alphas.size, 2))
        reg_params[:, 0] = alphas
        reg_params[:, 1] = l1
        results['reg_params'] = reg_params
        return results

class UoILasso():

    @classmethod
    def run(self, X, y, args):

        if 'comm' in list(args.keys()):
            comm = args['comm']
            rank = comm.rank
        else:
            comm = None
            rank = 0

        n_alphas = args['n_alphas']

        uoi = UoI_Lasso(
            fit_intercept=False,
            n_boots_sel=int(args['n_boots_sel']),
            n_boots_est=int(args['n_boots_est']),
                estimation_score=args['est_score'],
            stability_selection = args['stability_selection'],
            n_lambdas = n_alphas,
            comm = comm, 
            solver = 'pyc'
            )

        uoi.fit(X, y)
        
        results = {}
        results['coefs'] = uoi.supports_

        return results

class UoIMCP():

    @classmethod
    def run(self, X, y, args):

        if 'comm' in list(args.keys()):
            comm = args['comm']
            rank = comm.rank
        else:
            comm = None
            rank = 0

        n_alphas = args['n_alphas']

        uoi = UoI_MCP(
            fit_intercept=False,
            n_boots_sel=int(args['n_boots_sel']),
            n_boots_est=int(args['n_boots_est']),
                estimation_score=args['est_score'],
            stability_selection = args['stability_selection'],
            n_lambdas = n_alphas,
            comm = comm
            )

        uoi.fit(X, y)
        
        results = {}
        results['coefs'] = uoi.supports_

        return results


# Same class can be used for both MCP and SCAD based on our implementation
# of PycassoCV
class PYC():

    @classmethod
    def run(self, X, y, args):
        gamma = np.array(args['gamma'])
        penalty = args['penalty']
        n_alphas = args['n_alphas']

        estimator = PycassoGrid(penalty=penalty, n_alphas = n_alphas, 
                                fit_intercept=False, gamma=gamma)
        estimator.fit(X, y)
        reg_params = np.zeros((self.gamma.size * self.n_alphas, 2))

        for i, gamma in enumerate(self.gamma):
   
            reg_params[i * self.n_alphas:(i + 1) * self.n_alphas, 0] = gamma
            reg_params[i * self.n_alphas:(i + 1) * self.n_alphas, 1] = estimator.alphas

        results = {}
        results['coefs'] = estimators.coef_
        results['reg_params'] = reg_params

        return results


# Run R to solve slope
class SLOPE():

    @classmethod
    def run(self, X, y, args):

        lambda_method = args['lambda_method']
        lambda_args = args['lambda_args']
        
        estimates = np.zeros((self.lambda_args.size, n_features))
        for i, fdr in enumereate(lambda_args):
            slope = SLOPE(lambda_method='FDR', lambda_spec=fdr)
            slope.fit(X, y)
            estimates[i, :] = slope.coef_

        results = {}
        results['coefs'] = estimates

        return results
