import numpy as np
import pdb
import itertools
import time

from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from pyuoi.linear_model.cassolasso import UoI_Lasso
from pyuoi.linear_model.cassolasso import PycassoLasso
from pyuoi.linear_model.elasticnet import UoI_ElasticNet
from pyuoi.lbfgs import fmin_lbfgs

from mpi_utils.ndarray import Gatherv_rows

from info_criteria import GIC, eBIC
from utils import selection_accuracy

class Selector():

    def __init__(self, selection_method = 'CV'): 

        self.selection_method = selection_method

    def select(self, solutions, reg_params, *args):
        # Deal to the appropriate sub-function based on 
        # the provided selection method string

        oracle_penalty = -1

        if self.selection_method == 'CV':
            self.CV_selector(solutions, reg_params, *args)
        else:
            if self.selection_method in ['BIC', 'AIC']: 
                sidx = self.GIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'mBIC':
                sidx = self.mBIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'eBIC':
                sidx = self.eBIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'OIC':
                sidx, oracle_penalty = self.OIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'aBIC':
                sidx = self.aBIC_selector(solutions, reg_params, *args)

            # Select the coefficients and reg_params with lowest score
            coefs = solutions[sidx, :]
            selected_reg_param = reg_params[sidx]

            return coefs, selected_reg_param, oracle_penalty

    # Assumes the output of cross_validate
    def CV_selector(self, solutions, reg_params, 
                    test_scores, estimators, true_model):
        pass

    def GIC_selector(self, solutions, reg_params, X, y, true_model): 

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

        return np.argmin(scores)

    def mBIC_selector(self, solutions, reg_params, X, y, true_model):

        pass        

    def eBIC_selector(self, solutions, reg_params, X, y, true_model):

        n_features, n_samples = X.shape
        n_reg_params = len(reg_params)

        # Should be of the shape n_reg_params by n_samples
        y_pred = solutions @ X.T

        scores = np.array([eBIC(y.ravel(), y_pred[i, :], n_features,
                                np.count_nonzero(solutions[i, :]))
                                for i in range(n_reg_params)])

        return np.argmin(scores)

    def OIC_selector(self, solutions, reg_params, X, y, true_model): 

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

        oracle_penalty = penalties[np.argmax(selection_accuracies)]
        selection_idx = selection_idxs[np.argmax(selection_accuracies)]
        # Also return the maximum attainable selection accuracy

        return selection_idx, oracle_penalty

    def aBIC_selector(self, solutions, reg_params, X, y, true_model):

        # Do the whole procedure...
        pass


class UoISelector(Selector):

    def __init__(self, selection_method = 'CV'):

        super(UoISelector, self).__init__(selection_method)

    # Perform the UoI Union operation (median)
    def union(self, estimates, selected_idxs):

        selected_coefs = estimates[np.arange(estimates.shape[0]), selected_idxs, :]
        coefs = np.median(selected_coefs, axis = 0)
        return coefs

    def select(self, *args): 

        oracle_penalties = -1

        # For UoI, interpret CV selector to mean r2
        if self.selection_method == 'CV':
            coefs = self.r2_selector(*args)

        elif self.selection_method in ['BIC', 'AIC']: 
            coefs = self.GIC_selector(*args)
        elif self.selection_method == 'mBIC':
            coefs = self.mBIC_selector(*args)
        elif self.selection_method == 'eBIC':
            coefs = self.eBIC_selector(*args)
        elif self.selection_method == 'OIC':
            coefs, oracle_penalties = self.OIC_selector(*args)
        elif self.selection_method == 'aBIC':
            coefs = self.aBIC_selector(*args)

        return coefs, oracle_penalties

    def r2_selector(self, solutions, X, y, boots, true_model):

        # UoI Estimates have shape (n_boots_est, n_supports, n_coef)

        n_boots, n_supports, n_coefs = solutions.shape
        scores = np.zeros((n_boots, n_supports))

        for boot in range(n_boots):
            # Test data
            xx = X[boots[1][boot], :]
            yy = y[boots[1][boot]]
            y_pred = solutions[boot, ...] @ xx.T

            scores[boot, :] = np.array([r2_score(yy, y_pred[j, :]) for j in range(n_supports)])

        selected_idxs = np.argmax(scores, axis = 1)

        coefs = self.union(solutions, selected_idxs)
        return coefs

    def GIC_selector(self, solutions, X, y, boots, true_model):

        n_boots, n_supports, n_coefs = solutions.shape
        selected_idxs = np.zeros(n_boots, dtype = np.int)

        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]

            selected_idxs[boot] = super(UoISelector, self).GIC_selector(solutions[boot, ...],
                                                                     np.arange(n_supports),
                                                                     xx, yy, true_model)
        coefs = self.union(solutions, selected_idxs)
        return coefs

    def mBIC_selector(self, solutions, X, y, boots, true_model):
        pass

    def eBIC_selector(self, solutions, X, y, boots, true_model):

        n_boots, n_supports, n_coefs = solutions.shape
        selected_idxs = np.zeros(n_boots, dtype = np.int)

        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]

            selected_idxs[boot] = super(UoISelector, self).eBIC_selector(solutions[boot, ...],
                                                                     np.arange(n_supports),
                                                                     xx, yy, true_model)
        coefs = self.union(solutions, selected_idxs)
        return coefs

    def OIC_selector(self, solutions, X, y, boots, true_model):
        
        n_boots, n_supports, n_coefs = solutions.shape
        selected_idxs = np.zeros(n_boots, dtype = np.int)
        oracle_penalties = np.zeros(n_boots)
        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]

            sidx, op = super(UoISelector, self).OIC_selector(solutions[boot, ...],
                                                                     np.arange(n_supports),
                                                                     xx, yy, true_model)
            selected_idxs[boot] = sidx
            oracle_penalties[boot] = op

        coefs = self.union(solutions, selected_idxs)
        return coefs, oracle_penalties


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
        true_model = args['betas'].ravel()
        for selection_method in selection_methods:
            self.fit_and_select(X, y.ravel(), selection_method, true_model)
        
        del self.fitted_estimator
        return self.results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model): 

        # For cross validation, use the existing solution: 
        if selection_method == 'CV': 
            lasso = LassoCV(cv = self.cv_splits, 
                            alphas = self.alphas).fit(X, y.ravel())
            self.results[selection_method]['coefs'] = lasso.coef_
            self.results[selection_method]['reg_param'] = lasso.alpha_
            self.results[selection_method]['oracle_penalty'] = -1

        else: 
            if not hasattr(self, 'fitted_estimator'):
                # If not yet fitted, run the pycasso lasso

                lasso = PycassoLasso()
                lasso.fit(X, y, self.alphas)
                self.fitted_estimator = lasso
            # Extract the solution paths and regularization parameters 
            # and feed into the selector
            selector = Selector(selection_method = selection_method)
            coefs, reg_param, ops = selector.select(self.fitted_estimator.coef_, 
                                    self.alphas, X, y, true_model)
            self.results[selection_method]['coefs'] = coefs
            self.results[selection_method]['reg_param'] = reg_param
            self.results[selection_method]['oracle_penalty'] = ops

class UoILasso():

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):

        if 'comm' in list(args.keys()):
            comm = args['comm']
            rank = comm.rank
        else:
            comm = None
            rank = 0

        # Container for results
        self.results = {selection_method : {} 
                        for selection_method in selection_methods}

        for selection_method in selection_methods:
            self.fit_and_select(args, comm, rank, X, y.ravel(), selection_method)

        # Make sure to delete the fitted estimator, otherwise on subsequent calls
        # we won't end up re-fitting to the new data!
        del self.fitted_estimator
        return self.results

    @classmethod 
    def fit_and_select(self, args, comm, rank,
                       X, y, selection_method):

        if not hasattr(self, 'fitted_estimator'):
            
            # If not yet fitted, run the pycasso lasso
            uoi = UoI_Lasso(
                n_boots_sel=int(args['n_boots_sel']),
                n_boots_est=int(args['n_boots_est']),
                estimation_score=args['est_score'],
                stability_selection = args['stability_selection'],
                comm = comm
                )
            
            print('Fitting!')
            uoi.fit(X, y.ravel())
            self.fitted_estimator = uoi

            # Gather bootstrap information, which is currently 
            # distributed
            
            train_boots = np.array([uoi.boots[k][0] for k in uoi.boots.keys()])
            test_boots = np.array([uoi.boots[k][1] for k in uoi.boots.keys()])

            train_boots = Gatherv_rows(train_boots, comm = comm, root = 0)
            test_boots = Gatherv_rows(test_boots, comm = comm, root = 0)
            
            self.boots = [train_boots, test_boots]

        # Use the fact that UoI stores all of its estimates to
        # manually go in and select models and then take the union
        # using each distinct estimation score

        if rank == 0:
            true_model = args['betas'].ravel()
            selector = UoISelector(selection_method = selection_method)
            
            coefs, ops = selector.select(self.fitted_estimator.estimates_, X, y, 
                                         self.boots, true_model)
            self.results[selection_method]['coefs'] = coefs
            self.results[selection_method]['reg_param'] = -1
            self.results[selection_method]['oracle_penalty'] = np.mean(ops)
            # Make sure to store the oracle penalty somewhere
        else:
            self.results = None

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
