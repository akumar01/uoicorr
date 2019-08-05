import numpy as np
import pdb
import itertools
import time

from info_criteria import GIC, eBIC
from aBIC import aBIC


class Selector():

    def __init__(self, selection_method = 'BIC'): 

        self.selection_method = selection_method

    def select(self, solutions, reg_params, X, y, true_model, intercept=0):

        # Common operations to all
        n_features, n_samples = X.shape

        y_pred = solutions @ X.T + intercept

        # Deal to the appropriate sub-function based on 
        # the provided selection method string

        if self.selection_method in ['BIC', 'AIC']: 
            sdict = self.GIC_selector(y, y_pred, solutions, reg_params, 
                              n_features, n_samples)
        elif self.selection_method == 'mBIC':
            sdict = self.mBIC_selector()
        elif self.selection_method == 'eBIC':
            sdict = self.eBIC_selector(y, y_pred, solutions, reg_params, n_features)
        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(X, y, y_pred, solutions, reg_params, true_model)
        else:
            raise ValueError('Incorrect selection method specified')

        return sdict

    def GIC_selector(self, y, y_pred, solutions, reg_params, n_features, n_samples): 

        if self.selection_method == 'BIC':
            penalty = np.log(n_samples)
        if self.selection_method == 'AIC':
            penalty = 2
        
        scores = np.array([GIC(y.ravel(), y_pred[i, :], np.count_nonzero(solutions[i, :]),
                                penalty) for i in range(solutions.shape[0])])

        sidx = np.argmin(scores)

        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]
        return sdict

    def mBIC_selector(self):

        pass        

    def eBIC_selector(self, y, y_pred, solutions, reg_params, n_features):

        scores = np.array([eBIC(y.ravel(), y_pred[i, :], n_features,
                                np.count_nonzero(solutions[i, :]))
                                for i in range(solutions.shape[0])])
        sidx = np.argmin(scores)

        # Selection dict: Return coefs and selected_reg_param
        sdict = {}
        sdict['coefs'] = solutions[sidx, :]
        sdict['reg_param'] = reg_params[sidx]
        return sdict

    def aBIC_selector(self, X, y, y_pred, solutions, reg_params, true_model):

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

    def select(self, X, y, true_model): 

        # Apply UoI pre-processing
        X, y, = self.uoi._pre_fit(X, y)

        # For UoI, interpret CV selector to mean r2
        if self.selection_method == 'CV':
            sdict = self.r2_selector(X, y)
        elif self.selection_method in ['BIC', 'AIC']: 
            sdict = self.GIC_selector(X, y)
        elif self.selection_method == 'mBIC':
            sdict = self.mBIC_selector(X, y, *args)
        elif self.selection_method == 'eBIC':
            sdict = self.eBIC_selector(X, y, *args)
        elif self.selection_method == 'aBIC':
            sdict = self.aBIC_selector(X, y, *args)
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

    def GIC_selector(self, X, y):

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

            sdict_ = super(UoISelector, self).GIC_selector(yy, y_pred, solutions[boot, ...], 
                                                           np.arange(n_supports), 
                                                           n_features, n_samples)

            selected_coefs[boot, :] = sdict_['coefs']

        assert(np.allclose(selected_coefs, self.uoi.estimates_[np.arange(48), self.uoi.rp_max_idx_]))
        coefs = self.union(selected_coefs)
        sdict = {}
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

            sdict_ = super(UoISelector, self).eBIC_selector(solutions[boot, ...],
                                                            np.arange(n_supports),
                                                            xx, yy, true_model)
            selected_coefs[boot, :] = sdict_['coefs'] 

        coefs = self.union(selected_coefs)
        sdict = {}
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

            sdict_ = super(UoISelectsdict_or, self).aBIC_selector(solutions[boot, ...],
                                                                  np.arange(n_supports),
                                                                  xx, yy, true_model)
            bselected_coefs[boot, :] = sdict_['coefs'] 
            oselected_coefs[boot, :] = sdict_['oracle_coefs']
            bayesian_penalties[boot] = sdict_['bayesian_penalty']
            oracle_penalties[boot] = sdict_['oracle_penalty']

        coefs = self.union(selected_coefs)
        oracle_coefs = self.union(oselected_coefs)
        sdict = {}
        sdict['coefs'] = coefs
        sdict['oracle_coefs'] = oracle_coefs
        sdict['bayesian_penalties'] = bayesian_penalties
        sdict['oracle_penalties'] = oracle_penalties

        return sdict
