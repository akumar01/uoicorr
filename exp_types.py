import numpy as np
import pdb
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from pyuoi.linear_model.lasso import UoI_Lasso
from pyuoi.linear_model.cassolasso import PycassoLasso
from pyuoi.linear_model.elasticnet import UoI_ElasticNet
from gtv import GraphTotalVariance
import itertools
import time
from pyuoi.mpi_utils import Gatherv_rows
from pyuoi.lbfgs import fmin_lbfgs

class Selector():

    def __init__(self, selection_method = 'CV'): 

        self.selection_method = selection_method

    def select(self, *args):

        # Deal to the appropriate sub-function based on 
        # the provided selection method string
        if self.selection_method == 'CV':
            self.CV_selector(*args)
        # Straightforward scoring functions
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

        self.coefs = solutions[self.selection_idx, :]
        self.reg_param = reg_params[self.selection_idx]


    # Assumes the output of cross_validate
    def CV_selector(self, test_scores, estimators, 
                    solutions, reg_params, true_model):
        pass

    def GIC_selector(self, X, y, solutions, reg_params, true_model): 

        n_features, n_samples = X.shape
        n_reg_params = len(reg_params)

        if self.selection_method == 'BIC':
            penalty = np.log(samples)
        if self.selection_method == 'AIC':
            penalty = 2
        
        # Should be of the shape n_reg_params by n_samples
        y_pred = solutions @ X.T

        scores = np.array([GIC(y.ravel(), y_pred[i, :], np.count_nonzero(solutions[i, :],
                                penalty)) for i in range(n_reg_params)])

        # Select the coefficients and reg_params with lowest score
        self.selection_idx = int(np.argmin(self.scores))

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
        self.selection_idx = int(np.argmin(self.scores))


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
        selection_idxs = np.zeros(penalties.size)
        for i, penalty in enumerate(penalties):

            scores = np.array([GIC(y.ravel(), y_pred[j, :], 
                               np.count_nonzero(solutions[j, :], penalty)) 
                               for j in range(n_reg_params)])

            selection_idxs[i] = np.argmin(self.scores)
            selection_accuracies[i] = selection_accuracy(true_model, 
                                      solutions[selection_idx, :])

        self.oracle_penalty = penalties[np.argmax(selection_accuracies)]
        self.selection_idx = selection_idxs[np.argmax(selection_accuracies)]

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

        if 'forward_selection' in list(args.keys()):
            forward_selection = args['forward_selection']
        else:
            forward_selection = False

        if 'manual_penalty' in list(args.keys()):
            manual_penalty = args['manual_penalty']
        else:
            manual_penalty = 2

        if 'ss' in list(args.keys()):
            noise_level = args['ss']
        else:
            noise_level = 0

        uoi = UoI_Lasso(
            normalize=False,
            n_boots_sel=int(args['n_boots_sel']),
            n_boots_est=int(args['n_boots_est']),
            estimation_score=args['est_score'],
            stability_selection = args['stability_selection'],
            manual_penalty = manual_penalty,
            noise_level = noise_level,
            comm = comm
            )

        uoi.fit(X, y.ravel())

        return uoi

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

class GTV():

    @classmethod
    def run(self, X, y, args, groups = None):
        print('started run')
        cv_splits = 5
        lambda_S = np.linspace(0, 1, 10)
        lambda_TV = np.linspace(0, 1, 10)
        lambda_1 = np.linspace(0, 1, 10)

        cov = args['sigma']

        if not isinstance(lambda_S, np.ndarray):
            if np.isscalar(lambda_S):
                lambda_S = np.array([lambda_S])
            else:
                lambda_S = np.array(lambda_S)

        if not isinstance(lambda_TV, np.ndarray):
            if np.isscalar(lambda_TV):
                lambda_TV = np.array([lambda_TV])
            else:
                lambda_TV = np.array(lambda_TV)

        if not isinstance(lambda_1, np.ndarray):
            if np.isscalar(lambda_1):
                lambda_1 = np.array([lambda_1])
            else:
                lambda_1 = np.array(lambda_1)

        if 'comm' in list(args.keys()):
            comm = args['comm']
        else:
            comm = None

        if 'use_skeleton' in list(args.keys()):
            use_skeleton = args['use_skeleton']
        else:
            use_skeleton = True

        if 'threshold' in list(args.keys()):
            threshold = args['threshold']
        else:
            threshold = False

        scores = np.zeros((lambda_S.size, lambda_TV.size, lambda_1.size))

#        Use k-fold cross_validation
        kfold = KFold(n_splits = cv_splits, shuffle = True)

        # Parallelize hyperparameter search
        hparamlist = list(itertools.product(lambda_S, lambda_TV, lambda_1))

        if comm is not None:
            numproc = comm.Get_size()
            rank = comm.rank
            chunk_hparamlist = np.array_split(hparamlist, numproc)
            chunk_idx = rank
            num_tasks = len(chunk_hparamlist[chunk_idx])
        else:
            numproc = 1
            rank = 0
            chunk_hparamlist = [hparamlist]
            chunk_idx = 0
            num_tasks = len(hparamlist)

        cv_scores = np.zeros(len(chunk_hparamlist[chunk_idx]))
        for i, hparam in enumerate(chunk_hparamlist[chunk_idx]):
            t0 = time.time()
            gtv = GraphTotalVariance(lambda_S = hparam[0], lambda_TV = hparam[1],
                                     lambda_1 = hparam[2], normalize=True,
                                     warm_start = False, use_skeleton = use_skeleton,
                                     threshold = threshold, minimizer = 'lbfgs')
            scores = np.zeros(cv_splits)
            fold_idx = 0
            for train_index, test_index in kfold.split(X):
                # Fits
                gtv.fit(X[train_index, :], y[train_index], cov)
                # Score
                scores[fold_idx] = r2_score(y[test_index], X[test_index] @ gtv.coef_)
                fold_idx += 1
            cv_scores[i] = np.mean(scores)
            print('hparam time: %f' % (time.time() - t0))
        # Gather scores across processes
        if comm is not None:
            cv_scores = Gatherv_rows(cv_scores, comm, root=0)
        if rank == 0 or comm is None:
            print('finished iterating')
            best_idx = np.argmax(cv_scores)
            best_hparam = hparamlist[best_idx]
            # Return GTV fit the best hparam
            model = GraphTotalVariance(lambda_S = best_hparam[0],
                                     lambda_TV = best_hparam[1],
                                     lambda_1 = best_hparam[2], normalize=True,
                                     warm_start = False, use_skeleton = use_skeleton,
                                     threshold = threshold, minimizer = 'lbfgs')
            model.fit(X, y, cov)
        else:
            model = None
        return model
