import numpy as np
import pdb
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from pyuoi.linear_model.lasso import UoI_Lasso
from pyuoi.linear_model.elasticnet import UoI_ElasticNet
import itertools
import time
from pyuoi.mpi_utils import Gatherv_rows

class CV_Lasso():

    @classmethod
    def run(self, X, y, args):

        n_alphas = args['n_alphas']
        cv_splits = 10

        scores = np.zeros(n_alphas)

        lasso = Lasso(normalize=True, warm_start = False)

        # Use 10 fold cross validation. Do this in a manual way to enable use of warm_start and custom parameter sweeps
        kfold = KFold(n_splits = cv_splits, shuffle = True)

        # Generate alphas to use
        alphas = _alpha_grid(X = X, y = y.ravel(), l1_ratio = 1, normalize = True, n_alphas = n_alphas)

        for a_idx, alpha in enumerate(alphas):

            lasso.set_params(alpha = alpha)

            cv_scores = np.zeros(cv_splits)
            # Cross validation splits into training and test sets
            for i, cv_idxs in enumerate(kfold.split(X, y)):
                lasso.fit(X[cv_idxs[0], :], y[cv_idxs[0]])
                cv_scores[i] = r2_score(y[cv_idxs[1]], lasso.coef_ @ X[cv_idxs[1], :].T)

            # Average together cross-validation scores
            scores[a_idx] = np.mean(cv_scores)

        # Select the model with the maximum score
        max_score_idx = np.argmax(scores.ravel())
        lasso.set_params(alpha = alphas[max_score_idx])
        lasso.fit(X, y.ravel())
        return [lasso]


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

        uoi = UoI_Lasso(
            normalize=True,
            n_boots_sel=int(args['n_boots_sel']),
            n_boots_est=int(args['n_boots_est']),
            estimation_score=args['est_score'],
            stability_selection = args['stability_selection'],
            comm = comm
            )

        uoi.fit(X, y.ravel())

        return [uoi]    

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
        return [uoi]    

class EN():

    @classmethod
    def run(self, X, y, args):
        print('Started run method')
        l1_ratios = args['l1_ratios']
        n_alphas = args['n_alphas']
        cv_splits = 10

        if not isinstance(l1_ratios, np.ndarray):
            if np.isscalar(l1_ratios):
                l1_ratios = np.array([l1_ratios])
            else:
                l1_ratios = np.array(l1_ratios)
                
        en = ElasticNet(normalize=True, warm_start = True)

        # Use 10 fold cross validation. Do this in a manual way to enable use of warm_start 
        # and custom parameter sweeps
        kfold = KFold(n_splits = cv_splits, shuffle = True)

        reg_params = []
        
        for l1_idx, l1_ratio in enumerate(l1_ratios):
            # Generate alphas to use
            alphas = _alpha_grid(X = X, y = y.ravel(), l1_ratio = l1_ratio, normalize = True, n_alphas = n_alphas)
            reg_params.extend([{'l1_ratio': l1_ratio, 'alpha': alpha} for alpha in alphas])
        reg_params = np.array(reg_params)

        scores = []

        for reg_param in reg_params:
            en.set_params(**reg_param)
            cv_scores = np.zeros(cv_splits)
            # Cross validation splits into training and test sets
            for i, cv_idxs in enumerate(kfold.split(X, y)):
                en.fit(X[cv_idxs[0], :], y[cv_idxs[0]])
                cv_scores[i] = r2_score(y[cv_idxs[1]], en.coef_ @ X[cv_idxs[1], :].T)

            scores.append(np.mean(cv_scores))

        # Select the model with the maximum score
        max_score_idx = np.argmax(np.array(scores).ravel())
        en.set_params(**reg_params[max_score_idx])
        en.fit(X, y.ravel())
            
        return [en]

class GTV():

    @classmethod
    def run(self, X, y, args):
        print('started run')
	cv_splits = 5
        lambda_S = args['reg_params']['lambda_S']
        lambda_TV = args['reg_params']['lambda_TV']
        lambda_1 = args['reg_params']['lambda_1']

        cov = args['cov']

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
            use_skeleton = False

        if 'threshold' in list(args.keys()):
            threshold = args['threshold']
        else:
            thresholds = False 

        scores = np.zeros((lambda_S.size, lambda_TV.size, lambda_1.size))

#        Use k-fold cross_validation
        kfold = KFold(n_splits = cv_splits, shuffle = True)
        models = []

        # Parallelize hyperparameter search
        hparamlist = list(itertools.product(lambda_S, lambda_TV, lambda_1))

        if comm is not None:
            numproc = comm.Get_size() 
            rank = comm.rank
            chunk_hparamlist = np.array_split(hparamlist, numproc)
            chunk_idx = rank
            num_tasks = len(chunk_param_list[chunk_idx])
        else:
            chunk_hparamlist = [hparamlist]
            chunk_idx = 0
            num_tasks = len(hparamlist)

        cv_scores = np.zeros(len(hparamlist))
        for i, hparam in enumerate(chunk_hparamlist[chunk_idx]):
	    t0 = time.time()
            gtv = GraphTotalVariance(lambda_S = hparam[0], lambda_TV = hparam[1], 
                                     lambda_1 = hparam[2], normalize=True, 
                                     warm_start = False, use_skeleton = use_skeleton,
                                     threshold = threshold, method = 'lbfgs')
            scores = np.zeros(cv_splits)
            fold_idx = 0
            for train_index, test_index in kfold.split(X)
                # Fit
                gtv.fit(X[train_index, :], y[train_index], cov)
                # Score
                scores[fold_idx] = r2_score(y[test_index], X[test_index] @ gtv.coef_)
                fold_idx += 1
	    print('Process %d has finished iteration %d/%d in %f s' % (rank, i, numtasks, time.time() - t0))
            cv_scores[i] = np.mean(scores)
	# Gather scores across processes
	cv_scores = Gatherv_rows(cv_scores, comm, root=0)
	if rank == 0:
	    print('finished iterating')
	    best_idx = np.argmax(cv_scores)
	    best_hparam = hparam_list[best_idx]
            # Return GTV fit the best hparam
            model = GraphTotalVariance(lambda_S = best_hparam[0], 
                                 lambda_TV = best_hparam[1], 
                                 lambda_1 = best_hparam[2], normalize=True, 
                                 warm_start = False, use_skeleton = use_skeleton,
                                 threshold = threshold, method = 'lbfgs')
        	model.fit(X, y, cov)
	else:
	    model = None
        return model
