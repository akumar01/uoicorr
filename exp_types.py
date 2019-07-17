import numpy as np
import pdb
import itertools
import time

from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from pyuoi.linear_model.lasso import UoI_Lasso
from pyuoi.linear_model.cassolasso import PycassoLasso
from pyuoi.linear_model.elasticnet import UoI_ElasticNet
from pyuoi.mpi_utils import Gatherv_rows
from pyuoi.lbfgs import fmin_lbfgs

from gtv import GraphTotalVariance
from info_criteria import GIC, eBIC
from utils import selection_accuracy

class UoISelector(Selector):

    def __init__(self, selection_method = 'CV'):

        super(UoISelector, self).__init__(selection_method)

    def select(self, *args): 

        # For UoI, interpret CV selector to mean r2
        if self.selection_method == 'CV':
            self.r2_selector(*args)

        elif self.selection_method in ['BIC', 'AIC']: 
            self.GIC_selector(*args)
        elif self.selection_method == 'mBIC':
            self.mBIC_selector(*args)
        elif self.selection_method == 'eBIC':
            self.eBIC_selector(*args)
        elif self.selection_method == 'OIC':
            self.OIC_selector(*args)
        elif self.selection_method == 'aBIC':
            self.aBIC_selector(*args)

    def r2_selector(self, X, y, solutions, reg_params, true_model)

        # UoI Estimates have shape (n_boots_est, n_supports, n_coef)

        n_boots, n_supports, n_coefs = solutions.shape
        scores = np.zeros((n_boots, n_supports))
        for boot in n_boots:
            scores[boot, :] = 

class Selector():

    def __init__(self, selection_method = 'CV'): 

        self.selection_method = selection_method

    def select(self, *args):
        # Deal to the appropriate sub-function based on 
        # the provided selection method string
        if self.selection_method == 'CV':
            self.CV_selector(*args)

        elif self.selection_method in ['BIC', 'AIC']: 
            self.GIC_selector(*args)
        elif self.selection_method == 'mBIC':
            self.mBIC_selector(*args)
        elif self.selection_method == 'eBIC':
            self.eBIC_selector(*args)
        elif self.selection_method == 'OIC':
            self.OIC_selector(*args)
        elif self.selection_method == 'aBIC':
            self.aBIC_selector(*args)

    # Assumes the output of cross_validate
    def CV_selector(self, test_scores, estimators, 
                    solutions, reg_params, true_model):
        pass

    def GIC_selector(self, X, y, solutions, reg_params, true_model): 

        n_features, n_samples = X.shape
        n_reg_params = len(reg_params)

        if self.selection_method == 'BIC':
            penalty = np.log(n_samples)
        if self.selection_method == 'AIC':
            penalty = 2
        
        # Should be of the shape n_reg_params by n_samples
        y_pred = solutions @ X.T
        scores = np.array([GIC(y.ravel(), y_pred[i, :], np.count_nonzero(solutions[i, :]),
                                penalty) for i in range(n_reg_params)])

        # Select the coefficients and reg_params with lowest score
        self.selection_idx = np.argmin(scores)
        self.coefs = solutions[self.selection_idx, :]
        self.reg_param = reg_params[self.selection_idx]


    def mBIC_selector(self, X, y, solutions, reg_params, true_model):

        pass        

    def eBIC_selector(self, X, y, solutions, reg_params, true_model):

        n_features, n_samples = X.shape
        n_reg_params = len(reg_params)

        # Should be of the shape n_reg_params by n_samples
        y_pred = solutions @ X.T

        scores = np.array([eBIC(y.ravel(), y_pred[i, :], n_features,
                                np.count_nonzero(solutions[i, :]))
                                for i in range(n_reg_params)])

        # Select the coefficients and reg_params with lowest score
        self.selection_idx = int(np.argmin(scores))
        self.coefs = solutions[self.selection_idx, :]
        self.reg_param = reg_params[self.selection_idx]


    def OIC_selector(self, X, y, solutions, reg_params, true_model): 

        # Calculate the score for each penalty and assess the selection 
        # accuracy vs. the true model to find the oracle penalty
        n_features, n_samples = X.shape
        n_reg_params = len(reg_params)

        # Should be of the shape n_reg_params by n_samples
        y_pred = solutions @ X.T

        # Fix the penalty to be between 0 and 4 times the BIC
        penalties = np.linspace(0, 4 * np.log(n_samples), 50)
        selection_accuracies = np.zeros(penalties.size)
        selection_idxs = np.zeros(penalties.size, dtype = np.int32)
        for i, penalty in enumerate(penalties):

            scores = np.array([GIC(y.ravel(), y_pred[j, :], 
                               np.count_nonzero(solutions[j, :]), penalty) 
                               for j in range(n_reg_params)])

            selection_idxs[i] = np.argmin(scores)
            selection_accuracies[i] = selection_accuracy(true_model, 
                                      solutions[selection_idxs[i], :])

        self.oracle_penalty = penalties[np.argmax(selection_accuracies)]
        self.selection_idx = selection_idxs[np.argmax(selection_accuracies)]
        self.coefs = solutions[self.selection_idx, :]
        self.reg_param = reg_params[self.selection_idx]

    def aBIC_selector(self, X, y, solutions, reg_params, true_model):

        # Do the whole procedure...
        pass


class CV_Lasso():

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):

        self.n_alphas = args['n_alphas']
        self.cv_splits = 5

        # Draw alphas using the _alpha_grid method
        self.alphas = _alpha_grid(X, y.ravel(), n_alphas = self.n_alphas)

        # Container for results
        self.results = {selection_method : {} 
                        for selection_method in selection_methods}

        # For each selection method, obtain coefficients and selected 
        # hyperparameter
        for selection_method in selection_methods:
            self.fit_and_select(X, y.ravel(), selection_method, args['beta'])

        return self.results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model): 

        # For cross validation, use the existing solution: 
        if selection_method == 'CV': 
            lasso = LassoCV(cv = self.cv_splits, 
                            alphas = self.alphas).fit(X, y.ravel())
            self.results[selection_method]['coefs'] = lasso.coef_
            self.results[selection_method]['reg_params'] = lasso.alpha_

        else: 
            if not hasattr(self, 'fitted_estimator'):
                # If not yet fitted, run the pycasso lasso

                lasso = PycassoLasso()
                lasso.fit(X, y, self.alphas)
                self.fitted_estimator = lasso
            # Extract the solution paths and regularization parameters 
            # and feed into the selector
            selector = Selector(selection_method = selection_method)
            selector.select(X, y, self.fitted_estimator.coef_, 
                            self.alphas, true_model)
            self.results[selection_method]['coefs'] = selector.coefs
            self.results[selection_method]['reg_param'] = selector.reg_param

class UoILasso():

    @classmethod
    def run(self, X, y, args):

        if 'comm' in list(args.keys()):
            comm = args['comm']
        else:
            comm = None

        # Container for results
        self.results = {selection_method : {} 
                        for selection_method in selection_methods}

        for selection_method in selection_methods:
            self.fit_and_select(args, comm, X, y.ravel(), selection_method)

        return self.results

    @classmethod 
    def fit_and_select(self, args, comm, 
                       X, y, selection_method):

        if not hasattr(self, 'fitted_estimator'):
            # If not yet fitted, run the pycasso lasso
            uoi = UoI_Lasso(
                normalize=False,
                n_boots_sel=int(args['n_boots_sel']),
                n_boots_est=int(args['n_boots_est']),
                estimation_score=args['est_score'],
                stability_selection = args['stability_selection'],
                comm = comm
                )

            uoi.fit(X, y.ravel())
            self.fitted_estimator = uoi

        # Use the fact that UoI stores all of its estimates to
        # manually go in and select models and then take the union
        # using each distinct estimation score

        if comm.rank == 0:
            true_model = args['beta'].ravel()
            selector = UoISelector(selection_method = selection_method)
            selector.select(X, y, self.fitted_estimator, true_model)
            self.results[selection_method]['coefs'] = selector.coefs
            self.results[selection_method]['reg_param'] = selector.reg_param            
            # Make sure to store the oracle penalty somewhere

class UoIElasticNet():

    @classmethod
    def run(self, X, y, args):

        l1_ratios = args['l1_ratios']

        # Ensure that the l1_ratios are an np array
        if not isinstance(l1_ratios, np.ndarray):
            if np.isscalar(l1_ratios):
                l1_ratios = np.array([l1_ratios])
            else:
                l1_ratios = np.array(l1_ratios)

        if 'comm' in list(args.keys()):
            comm = args['comm']
        else:
            comm = None

        uoi = UoI_ElasticNet(
            normalize=True,
            n_boots_sel=int(args['n_boots_sel']),
            n_boots_est=int(args['n_boots_est']),
            alphas = l1_ratios,
            estimation_score=args['est_score'],
            warm_start = False,
            stability_selection=args['stability_selection'],
            comm = comm
        )

        uoi.fit(X, y.ravel())
        return uoi

class EN():

    @classmethod
    def run(self, X, y, args, groups = None):
        l1_ratios = args['l1_ratios']
        n_alphas = args['n_alphas']
        cv_splits = 5

        if not isinstance(l1_ratios, np.ndarray):
            if np.isscalar(l1_ratios):
                l1_ratios = np.array([l1_ratios])
            else:
                l1_ratios = np.array(l1_ratios)

        en = ElasticNetCV(cv = 5, n_alphas = 48,
                        l1_ratio = l1_ratios).fit(X, y.ravel())

        return en
