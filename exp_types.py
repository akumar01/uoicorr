import numpy as np
import pdb
import itertools
import time

from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV, ElasticNet
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from pyuoi.linear_model import UoI_Lasso
from pyuoi.linear_model.cassolasso import PycassoLasso
from pyuoi.linear_model import UoI_ElasticNet
from pyuoi.linear_model.casso_en import PycassoElasticNet
from pyuoi.lbfgs import fmin_lbfgs
from pyuoi.utils import log_likelihood_glm, BIC

from mpi_utils.ndarray import Gatherv_rows

from info_criteria import GIC, eBIC
from aBIC import aBIC
from utils import selection_accuracy
from pycasso_cv import PycassoCV, PycassoGrid

class Selector():

    def __init__(self, selection_method = 'CV'): 

        self.selection_method = selection_method

    def select(self, solutions, reg_params, *args):
        # Deal to the appropriate sub-function based on 
        # the provided selection method string

        if self.selection_method == 'CV':
            self.CV_selector(solutions, reg_params, *args)
        else:
            if self.selection_method in ['BIC', 'AIC']: 
                sdict = self.GIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'mBIC':
                sdict = self.mBIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'eBIC':
                sdict = self.eBIC_selector(solutions, reg_params, *args)
            elif self.selection_method == 'aBIC':
                sdict = self.aBIC_selector(solutions, reg_params, *args)

            return sdict

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


        sidx = np.argmin(scores)
        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]

        return sdict

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
        sidx = np.argmin(scores)
        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]

        return sdict

    def aBIC_selector(self, solutions, reg_params, X, y, true_model):

        oracle_penalty, bayesian_penalty, bidx, oidx = \
        aBIC(X, y, solutions, true_model)

        sidx = np.argmin(scores)
        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[bidx, :]
        sdict['reg_param'] = reg_params[bidx]
        sdict['oracle_coefs'] = solutions[oidx, :]
        sdict['oracle_penalty'] = oracle_penalty
        sdict['bayesian_penalty'] = bayesian_penalty

        return sdict

class UoISelector(Selector):

    def __init__(self, uoi, selection_method = 'CV'):

        super(UoISelector, self).__init__(selection_method)
        self.uoi = uoi

    # Perform the UoI Union operation (median)
    def union(self, selected_solutions):
        coefs = np.median(selected_solutions, axis = 0)
        return coefs

    def select(self, *args): 

        # For UoI, interpret CV selector to mean r2
        if self.selection_method == 'CV':
            sdict = self.r2_selector(*args)

        elif self.selection_method in ['BIC', 'AIC']: 
            sdict = self.GIC_selector(*args)
        elif self.selection_method == 'mBIC':
            sdict = self.mBIC_selector(*args)
        elif self.selection_method == 'eBIC':
            sdict = self.eBIC_selector(*args)
        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(*args)

        return sdict

    def r2_selector(self, X, y, boots, true_model):

        # UoI Estimates have shape (n_boots_est, n_supports, n_coef)

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        boots = self.uoi.boots

        n_boots, n_supports, n_coefs = solutions.shape
        scores = np.zeros((n_boots, n_supports))

        for boot in range(n_boots):
            # Test data
            xx = X[boots[1][boot], :]
            yy = y[boots[1][boot]]  
            y_pred = solutions[boot, ...] @ xx.T + intercepts[boot, :][:, np.newaxis]

            scores[boot, :] = np.array([r2_score(yy, y_pred[j, :]) for j in range(n_supports)])

        selected_idxs = np.argmax(scores, axis = 1)
        coefs = self.union(solutions, selected_idxs)

        # Return just the coefficients that result
        sdict['coefs'] = coefs

        return sdict

    def GIC_selector(self, X, y, boots, true_model):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_

        n_boots, n_supports, n_coefs = solutions.shape
        selected_coefs = np.zeros((n_boots, n_coefs))

        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]

            if self.selection_method == 'BIC':
                penalty = np.log(yy.shape[-1])
            if self.selection_method == 'AIC':
                penalty = 2

            # Should be of the shape n_reg_params by n_samples
            y_pred = solutions[boot, ...] @ xx.T + intercepts[boot, :][:, np.newaxis]
            sdict_ = np.array([GIC(yy, y_pred[i, :], np.count_nonzero(solutions[boot, i, :]),
                                    penalty) for i in range(n_supports)])

            selected_coefs[boot, :] = sdict_['coefs']

        coefs = self.union(selected_coefs)
        sdict['coefs'] = coefs

        return sdict

    def mBIC_selector(self, X, y, true_model):
        pass

    def eBIC_selector(self, X, y, true_model):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        boots = self.uoi.boots

        n_boots, n_supports, n_coefs = solutions.shape
        selected_coefs = np.zeros((n_boots, n_coefs))

        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]] - intercepts[boot, :]

            sdict_ = super(UoISelectsdict_or, self).eBIC_selector(solutions[boot, ...],
                                                                  np.arange(n_supports),
                                                                  xx, yy, true_model)
            
            selected_coefs[boot, :] = sdict_['coefs'] 

        coefs = self.union(selected_coefs)
        sdict['coefs'] = coefs
        return sdict

    def aBIC_selector(self, X, y, true_model):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        boots = self.uoi.boots

        n_boots, n_supports, n_coefs = solutions.shape
        bselected_coefs = np.zeros((n_boots, n_coefs))
        oselected_coefs = np.zeros((n_boots, n_coefs))
        bayesian_penalties = np.zeros(n_boots)
        oracle_penalties = np.zeros(n_boots)

        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]] - intercepts[boot, :]

            sdict_ = super(UoISelectsdict_or, self).eBIC_selector(solutions[boot, ...],
                                                                  np.arange(n_supports),
                                                                  xx, yy, true_model)
            bselected_coefs[boot, :] = sdict_['coefs'] 
            oselected_coefs[boot, :] = sdict_['oracle_coefs']
            bayesian_penalties[boot] = sdict_['bayesian_penalty']
            oracle_penalties[boot] = sdict_['oracle_penalty']

        coefs = self.union(selected_coefs)
        oracle_coefs = self.union(oselected_coefs)

        sdict['coefs'] = coefs
        sdict['oracle_coefs'] = oracle_coefs
        sdict['bayesian_penalties'] = bayesian_penalties
        sdict['oracle_penalties'] = oracle_penalties

        return sdict

class CV_Lasso():

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):
        
        if hasattr(self, 'fitted_estimator'):
            del self.fitted_estimator
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
        
        return self.results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model): 

        # For cross validation, use the existing solution: 
        if selection_method == 'CV': 
            lasso = LassoCV(cv = self.cv_splits, 
                            alphas = self.alphas).fit(X, y.ravel())
            self.results[selection_method]['coefs'] = lasso.coef_
            self.results[selection_method]['reg_param'] = lasso.alpha_

        else: 
            if not hasattr(self, 'fitted_estimator'):
                # If not yet fitted, run the pycasso lasso

                lasso = PycassoLasso()
                lasso.fit(X, y, self.alphas)
                self.fitted_estimator = lasso
            # Extract the solution paths and regularization parameters 
            # and feed into the selector
            selector = Selector(selection_method = selection_method)
            sdict = selector.select(self.fitted_estimator.coef_, 
                                    self.alphas, X, y, true_model)
            self.results[selection_method] = sdict

class EN(CV_Lasso):

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):

        # Run RidgeCV to determine L2 penalty
        rdge = RidgeCV(alphas = np.linspace(1e-5, 100, 500)).fit(X, y)
        self.l2 = rdge.alpha_
        results = super(EN, self).run(X, y, args, selection_methods)

        return results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model):

        _, n_features = X.shape

        # For cross validation, use the existing solution: 
        if selection_method == 'CV': 
            en = ElasticNetCV(cv = self.cv_splits, l1_ratio = self.l1_ratio, 
                            n_alphas = self.n_alphas).fit(X, y.ravel())
            self.results[selection_method]['coefs'] = en.coef_
            self.results[selection_method]['reg_param'] = [en.l1_ratio_, en.alpha_]

        else:

            # Fit elastic net to self-defined grid 
            if not hasattr(self, 'fitted_estimator'):
                estimates = np.zeros((self.alphas.size, n_features))
                print('Fitting!')

                for i, l1 in enumerate(self.alphas):
                    l1_ratio = l1/(l1 + 2 * self.l2)
                    alpha = l1 + 2 * self.l2

                    en = ElasticNet(l1_ratio = l1_ratio, alpha = alpha, 
                                    fit_intercept = False)
                    en.fit(X, y)
                    estimates[i, :] = (1 + self.l2) * en.coef_.ravel()

                reg_params = np.zeros((self.alphas.size, 2))
                reg_params[:, 0] = self.l2
                reg_params[:, 1] = self.alphas
                self.fitted_estimator = dummy_estimator(coefs = estimates, 
                                                reg_params = reg_params)

            selector = Selector(selection_method)
            sdict = selector.select(self.fitted_estimator.coefs, 
                                                    self.fitted_estimator.reg_params, X, y, true_model)

            self.results[selection_method] = sdict

class UoILasso():

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):

        if hasattr(self, 'fitted_estimator'):
            del self.fitted_estimator

        if 'comm' in list(args.keys()):
            comm = args['comm']
            rank = comm.rank
        else:
            comm = None
            rank = 0

        self.n_alphas = args['n_alphas']

        # Container for results
        self.results = {selection_method : {} 
                        for selection_method in selection_methods}

        for selection_method in selection_methods:
            self.fit_and_select(args, comm, rank, X, y.ravel(), selection_method)

        # Make sure to delete the fitted estimator, otherwise on subsequent calls
        # we won't end up re-fitting to the new data!
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
                n_lambdas = self.n_alphas,
                comm = comm
                )
            
            print('Fitting!')
            uoi.fit(X, y.ravel())
            self.fitted_estimator = uoi

            # Gather bootstrap information, which is currently 
            # distributed
            
            train_boots = np.array([uoi.boots[k][0] for k in uoi.boots.keys()])
            test_boots = np.array([uoi.boots[k][1] for k in uoi.boots.keys()])

            if comm is not None:
                train_boots = Gatherv_rows(train_boots, comm = comm, root = 0)
                test_boots = Gatherv_rows(test_boots, comm = comm, root = 0)
            
            self.boots = [train_boots, test_boots]

        # Use the fact that UoI stores all of its estimates to
        # manually go in and select models and then take the union
        # using each distinct estimation score

        if rank == 0:

            true_model = args['betas'].ravel()
            selector = UoISelector(self.fitted_estimator, selection_method = selection_method)
            
            sdict = selector.select(X, y, self.boots, true_model)
            self.results[selection_method] = sdict

        else:
            self.results = None

class UoIElasticNet(UoILasso):

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):

        super(UoIElasticNet, self).run(X, y, args, selection_methods)

        return self.results

    @classmethod
    def fit_and_select(self, args, comm, rank, X, y, selection_method):

        if not hasattr(self, 'fitted_estimator'):
            
            # If not yet fitted, run elastic net
            uoi = UoI_ElasticNet(
                n_boots_sel=int(args['n_boots_sel']),
                n_boots_est=int(args['n_boots_est']),
                estimation_score=args['est_score'],
                stability_selection = args['stability_selection'],
                n_lambdas = self.n_alphas,
                comm = comm
                )
            
            print('Fitting!')
            uoi.fit(X, y.ravel())
            self.fitted_estimator = uoi

            # Gather bootstrap information, which is currently 
            # distributed
            
            train_boots = np.array([uoi.boots[k][0] for k in uoi.boots.keys()])
            test_boots = np.array([uoi.boots[k][1] for k in uoi.boots.keys()])
            if comm is not None:
                train_boots = Gatherv_rows(train_boots, comm = comm, root = 0)
                test_boots = Gatherv_rows(test_boots, comm = comm, root = 0)
            
            self.boots = [train_boots, test_boots]

        # Use the fact that UoI stores all of its estimates to
        # manually go in and select models and then take the union
        # using each distinct estimation score

        if rank == 0:
            true_model = args['betas'].ravel()
            selector = UoISelector(self.fitted_estimator, selection_method = selection_method)
            
            sdict = selector.select(X, y, self.boots, true_model)
            self.results[selection_method] = sdict
        else:
            self.results = None

# Same class can be used for both MCP and SCAD based on our implementation
# of PycassoCV
class PYC(CV_Lasso):

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):
        self.gamma = np.array(args['gamma'])
        self.penalty = args['penalty']
        super(PYC, self).run(X, y, args, selection_methods)        
        return self.results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model): 

        _, n_features = X.shape
        penalty = self.penalty
        # For cross validation, use the existing solution: 
        if selection_method == 'CV': 
            estimator = PycassoCV(penalty = penalty, nfolds = self.cv_splits, 
                                  n_alphas = self.n_alphas, 
                                  fit_intercept=False)
            estimator.fit(X, y.ravel())
            self.results[selection_method]['coefs'] = estimator.coef_
            self.results[selection_method]['reg_param'] = \
                                        [estimator.gamma_, estimator.alpha_]
            self.results[selection_method]['oracle_penalty'] = -1

        else: 
            if not hasattr(self, 'fitted_estimator'):
                # If not yet fitted, run the pycasso lasso
                estimator = PycassoGrid(penalty = penalty, n_alphas = self.n_alphas,
                                        fit_intercept = False, gamma = self.gamma)
                print('Fitting!')
                estimator.fit(X, y)
                self.fitted_estimator = estimator

                # Ravel the grid

                coefs = self.fitted_estimator.coef_.reshape(-1, n_features)
                reg_params = np.zeros((self.gamma.size * self.n_alphas, 2))

                for i, gamma in enumerate(self.gamma):
           
                    reg_params[i * self.n_alphas:(i + 1) * self.n_alphas, 0] = gamma
                    reg_params[i * self.n_alphas:(i + 1) * self.n_alphas, 1] = estimator.alphas

                self.coef_ = coefs
                self.reg_params = reg_params

            # Extract the solution paths and regularization parameters 
            # and feed into the selector
            selector = Selector(selection_method = selection_method)
            sdict = selector.select(self.coef_, 
                                    self.reg_params, X, y, true_model)
            self.results[selection_method] = sdict

# Run R to solve slope
class SLOPE(CV_Lasso):

    @classmethod
    def run(self, X, y, args, selection_method = ['BIC']):

        self.lambda_method = args['lambda_method']
        self.reg_params = args['slope_reg_params']
        super(SLOPE, self).run(X, y, args, selection_methods)        
        return self.results

    def fit_and_select(self, X, y, selection_method, true_model):

        if selection_method == 'CV':
            raise NotImplementedError('CV not supported')

        if not hasattr(self, 'fitted_estimator'):

            n_samples, n_features = X.shape

            # Initialize R interface. Make sure SLOPE is installed! 
            slope = rpy2.robjects.packages.importr('SLOPE')
            rpy2.robjects.numpy2ri.activate()

            if self.lambda_method == 'FDR':
                lambdas = np.array([slope.create_lambda(n_samples, n_features, 
                                              fdr = fdr,
                                              method = 'gaussian')
                                    for fdr in self.reg_params['slopre_reg_params']['fdr']])
            elif self.lambda_method == 'user':
                lambdas = self.reg_params['slope_reg_params']['lambda']

            if lambdas.ndim == 1:
                lambdas = lambdas[np.newaxis, :]

            estimates = np.zeros((lambdas.shape[0], n_features))

            for i in range(lambdas.shape[0]):
                result = slope.SLOPE_solver(X, y, lambdas[i, :])
                estimates[i, :] = np.array(result[4])

            # Feed in dummy reg_params --> not to interested in them
            self.fitted_estimator = dummy_estimator(estimates, np.arange(lambdas.shape[0]))

            selector = Selector(selection_method = selection_method)

            sdict = selector.select(self.fitted_estimator.coefs, 
                                                    self.fitted_estimator.reg_params, 
                                                    X, y, true_model)

            self.results[selection_method]= sdict

# Convenience Class
class dummy_estimator():

    def __init__(self, coefs, reg_params):

        self.coefs = coefs
        self.reg_params = reg_params

