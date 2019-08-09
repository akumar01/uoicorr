import numpy as np
import pdb
import itertools
import time

from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import RidgeCV

from pyuoi.linear_model import UoI_Lasso
from pyuoi.linear_model import UoI_ElasticNet

from pyc_based.lm import PycassoLasso, PycassoElasticNet
from pyc_based.pycasso_cv import PycassoCV, PycassoGrid, PycEnCV

from r_based.slope import SLOPE as SLOPE_
from r_based.slope import SLOPE_CV

from selection import Selector, UoISelector

class StandardLM_experiment():

    @classmethod
    def run(self, X, y, args, selection_methods=['CV']):

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

class CV_Lasso(StandardLM_experiment):

    @classmethod
    def run(self, X, y, args, selection_methods = ['CV']):
        results = super(CV_Lasso, self).run(X, y, args, selection_methods)
        return results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model): 

        # Use the pycassocv for Lasso
        if selection_method == 'CV': 
            print('Fitting!')
            lasso = PycassoCV(penalty='l1', n_alphas=48, 
                              gamma=3, alphas=self.alphas)
            lasso.fit(X, y.ravel())
            # lasso = LassoCV(cv = self.cv_splits, 
            #                 alphas = self.alphas).fit(X, y.ravel())
            self.results[selection_method]['coefs'] = lasso.coef_
            self.results[selection_method]['reg_param'] = lasso.alpha_

        else: 
            if not hasattr(self, 'fitted_estimator'):
                # If not yet fitted, run the pycasso lasso
                print('Fitting!')
                lasso = PycassoLasso(alphas = self.alphas)
                lasso.fit(X, y)
                self.fitted_estimator = lasso
            # Extract the solution paths and regularization parameters 
            # and feed into the selector
            selector = Selector(selection_method = selection_method)
            sdict = selector.select(self.fitted_estimator.coef_, 
                                    self.alphas, X, y, true_model)
            self.results[selection_method] = sdict

class EN(StandardLM_experiment):

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

        # For cross validation, use our solution that uses pycasso
        if selection_method == 'CV': 
            print('Fitting!')
            en = PycEnCV(n_folds = self.cv_splits, fit_intercept=False, 
                         lambda1=self.alphas, lambda2=self.l2)
            en.fit(X, y.ravel())
            # en = ElasticNetCV(cv = self.cv_splits, l1_ratio = self.l1_ratio, 
            #                 n_alphas = self.n_alphas).fit(X, y.ravel())
            self.results[selection_method]['coefs'] = en.coef_
            self.results[selection_method]['reg_param'] = [en.lambda1_, en.lambda2_]
        else:

            if not hasattr(self, 'fitted_estimator'):
                print('Fitting!')

                en = PycassoElasticNet(fit_intercept = False, 
                                       lambda1=self.alphas, lambda2=self.l2)

                en.fit(X, y)
                estimates = en.coef_

                reg_params = np.zeros((self.alphas.size, 2))
                reg_params[:, 0] = self.l2
                reg_params[:, 1] = self.alphas
                self.fitted_estimator = en
            selector = Selector(selection_method)
            sdict = selector.select(self.fitted_estimator.coef_, 
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
            t0 = time.time()
            uoi = UoI_Lasso(
                fit_intercept=False,
                n_boots_sel=int(args['n_boots_sel']),
                n_boots_est=int(args['n_boots_est']),
                estimation_score=args['est_score'],
                stability_selection = args['stability_selection'],
                n_lambdas = self.n_alphas,
                comm = comm, 
                solver = 'cd'
                )
            if rank == 0:
                print('Fitting!')
            uoi.fit(X, y.ravel())
            if rank == 0:
                print('fit time: %f' % (time.time() - t0)) 
            self.fitted_estimator = uoi

        # Use the fact that UoI stores all of its estimates to
        # manually go in and select models and then take the union
        # using each distinct estimation score

        if rank == 0:

            true_model = args['betas'].ravel()
            selector = UoISelector(self.fitted_estimator, selection_method = selection_method)
            
            sdict = selector.select(X, y, true_model)
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
                fit_intercept=False,
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

        # Use the fact that UoI stores all of its estimates to
        # manually go in and select models and then take the union
        # using each distinct estimation score

        if rank == 0:
            true_model = args['betas'].ravel()
            selector = UoISelector(self.fitted_estimator, selection_method = selection_method)
            
            sdict = selector.select(X, y, true_model)
            self.results[selection_method] = sdict
        else:
            self.results = None

# Same class can be used for both MCP and SCAD based on our implementation
# of PycassoCV
class PYC(StandardLM_experiment):

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
class SLOPE(StandardLM_experiment):

    @classmethod
    def run(self, X, y, args, selection_methods = ['BIC']):

        # Hard code since we are using stale args
        self.lambda_method = 'FDR'
        # self.lambda_args = np.linspace(0.01, 0.9, 100)
        self.lambda_args = [0.2]
        self.lambda_args = np.array(self.lambda_args)
        super(SLOPE, self).run(X, y, args, selection_methods)        
        return self.results

    @classmethod
    def fit_and_select(self, X, y, selection_method, true_model):

        if selection_method == 'CV':

            slope = SLOPE_CV(nfolds = self.cv_splits, 
                             lambda_method=self.lambda_method, 
                             lambda_spec=self.lambda_args)
            slope.fit(X, y)
            self.results[selection_method]['coefs'] = slope.coef_
            # Store the FDR
            self.results[selection_method]['reg_param'] = slope.fdr_

        else:
            if not hasattr(self, 'fitted_estimator'):
     
                _, n_features = X.shape

                estimates = np.zeros((self.lambda_args.size, n_features))
                for i, fdr in enumerate(self.lambda_args):
                    slope = SLOPE_(lambda_method='FDR', lambda_spec=fdr)
                    slope.fit(X, y)
                    estimates[i, :] = slope.coef_

                self.fitted_estimator = dummy_estimator(coefs = estimates, 
                                                        reg_params = self.lambda_args)
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

