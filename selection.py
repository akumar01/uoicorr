import numpy as np
import pdb
import itertools
import time

from info_criteria import GIC, eBIC, gMDL, empirical_bayes
from aBIC import aBIC, mBIC
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class Selector():

    def __init__(self, selection_method = 'BIC'): 

        self.selection_method = selection_method

    def select(self, solutions, reg_params, X, y, true_model, intercept=0):

        # Common operations to all
        n_samples, n_features = X.shape

        # Fit OLS models
        # OLS_solutions = np.zeros(solutions.shape)

        # for i in range(solutions.shape[0]):
        #    support = solutions[i, :].astype(bool)
        #    if np.count_nonzero(solutions[i, :] > 0):
        #        linmodel = LinearRegression(fit_intercept=False)
        #        linmodel.fit(X[:, support], y)
        #        OLS_solutions[i, support] = linmodel.coef_

        # y_pred = OLS_solutions @ X.T + intercept
        y_pred = solutions @ X.T + intercept
        # Deal to the appropriate sub-function based on 
        # the provided selection method string

        if self.selection_method in ['mBIC', 'eBIC', 'BIC', 'AIC',
                                       'gMDL', 'empirical_bayes']:
            sdict = self.selector(X, y, y_pred, solutions, 
                                  reg_params)
        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(X, y, solutions, 
                                       reg_params, true_model)
        else:
            raise ValueError('Incorrect selection method specified')
        return sdict

    def selector(self, X, y, y_pred, solutions, reg_params):

        n_samples, n_features = X.shape
        sdict = {}
        if self.selection_method in ['AIC', 'BIC']:
            if self.selection_method == 'BIC':
                penalty = np.log(n_samples)
            if self.selection_method == 'AIC':
                penalty = 2
            
            scores = np.array([GIC(y.ravel(), y_pred[i, :], 
                               np.count_nonzero(solutions[i, :]),
                               penalty) for i in range(solutions.shape[0])])
            sidx = np.argmin(scores)
        if self.selection_method == 'mBIC':
            scores = mBIC(X, y, solutions)
            sidx = np.argmax(scores)
        elif self.selection_method ==  'eBIC':
            scores = np.array([eBIC(y.ravel(), y_pred[i, :], n_features,
                                    np.count_nonzero(solutions[i, :]))
                                    for i in range(solutions.shape[0])])
            sidx = np.argmin(scores)
        elif self.selection_method == 'gMDL':
            scores = np.array([gMDL(y.ravel(), y_pred[i, :],
                        np.count_nonzero(solutions[i, :]))
                        for i in range(solutions.shape[0])])
            sidx = np.argmin(scores[:, 0])
            sdict['effective_penalty'] = scores[sidx, 1] 
        elif self.selection_method == 'empirical_bayes':
            scores = np.array([empirical_bayes(X, y,
                                   solutions[i, :])
                   for i in range(solutions.shape[0])])
            sidx = np.argmin(scores[:, 0])
            sdict['effective_penalty'] = scores[sidx, 1]
        # Selection dict: Return coefs and selected_reg_param
        
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]
        return sdict

    def aBIC_selector(self, X, y, solutions, reg_params, true_model):

        oracle_penalty, bayesian_penalty, bidx, oidx, spest = \
        aBIC(X, y, solutions, true_model)

        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[bidx, :]
        sdict['reg_param'] = reg_params[bidx]
        sdict['oracle_coefs'] = solutions[oidx, :]
        sdict['oracle_penalty'] = oracle_penalty
        sdict['effective_penalty'] = bayesian_penalty
        sdict['sparsity_estimates'] = spest

        return sdict

class UoISelector(Selector):

    def __init__(self, uoi, selection_method = 'CV'):

        super(UoISelector, self).__init__(selection_method)
        self.uoi = uoi

    # Perform the UoI Union operation (median)
    def union(self, selected_solutions):
        coefs = np.median(selected_solutions, axis = 0)
        return coefs

    def select(self, X, y, true_model): 

        # Apply UoI pre-processing
        X, y, = self.uoi._pre_fit(X, y)

        # For UoI, interpret CV selector to mean r2
        if self.selection_method == 'CV':
            sdict = self.r2_selector(X, y)
        elif self.selection_method in ['BIC', 'AIC', 'mBIC',
                                       'eBIC', 'gMDL', 'empirical_bayes']: 
            sdict = self.selector(X, y)
        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(X, y, true_model)
        else:
            raise ValueError('Incorrect selection method specified')

        # Apply UoI post-processing (copy and pasted)
        if self.uoi.standardize:
            sX = self.uoi._X_scaler
            sy = self.uoi._y_scaler
            sdict['coefs'] /= sX.scale_
            sdict['coefs'] *= sy.scale_

        return sdict

    def r2_selector(self, X, y):

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
        coefs = self.union(solutions[np.arange(n_boots), selected_idxs])

        # Return just the coefficients that result
        sdict = {}
        sdict['scores'] = scores
        sdict['coefs'] = coefs

        return sdict

    def selector(self, X, y):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        boots = self.uoi.boots
        n_boots, n_supports, n_coefs = solutions.shape
        selected_coefs = np.zeros((n_boots, n_coefs))

        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]
            n_samples, n_features = xx.shape
            y_pred = solutions[boot, ...] @ xx.T + intercepts[boot, :][:, np.newaxis]

            sdict_ = super(UoISelector, self).selector(yy, y_pred, solutions[boot, ...], 
                                                           np.arange(n_supports)) 

            selected_coefs[boot, :] = sdict_['coefs']

        coefs = self.union(selected_coefs)
        sdict = {}
        sdict['coefs'] = coefs

        return sdict

    def aBIC_selector(self, X, y, true_model):

        solutions = self.uoi.estimates_
        intercepts = self.uoi.intercepts_
        assert(np.all(intercepts == 0))
        boots = self.uoi.boots

        n_boots, n_supports, n_coefs = solutions.shape
        bselected_coefs = np.zeros((n_boots, n_coefs))
        oselected_coefs = np.zeros((n_boots, n_coefs))
        bayesian_penalties = np.zeros(n_boots)
        oracle_penalties = np.zeros(n_boots)

        sparsity_estimates = []
        for boot in range(n_boots):

            # Train data
            xx = X[boots[0][boot], :]
            yy = y[boots[0][boot]]

            sdict_ = super(UoISelector, self).aBIC_selector(xx, yy, solutions[boot, ...],
                                                            np.arange(n_supports),
                                                            true_model)
            bselected_coefs[boot, :] = sdict_['coefs'] 
            oselected_coefs[boot, :] = sdict_['oracle_coefs']
            bayesian_penalties[boot] = sdict_['bayesian_penalty']
            oracle_penalties[boot] = sdict_['oracle_penalty']
            sparsity_estimates.append(sdict_['sparsity_estimates'])

        coefs = self.union(bselected_coefs)
        oracle_coefs = self.union(oselected_coefs)
        sdict = {}
        sdict['coefs'] = coefs
        sdict['oracle_coefs'] = oracle_coefs
        sdict['effective_penalty'] = bayesian_penalties
        sdict['oracle_penalty'] = oracle_penalties
        sdict['sparsity_estimates'] = np.array(sparsity_estimates, dtype = float)

        return sdict
