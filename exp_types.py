import numpy as np
import pdb
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from pyuoi.linear_model.lasso import UoI_Lasso
from pyuoi.linear_model.elasticnet import UoI_ElasticNet
from gtv import GraphTotalVariance
import itertools
import time
from pyuoi.mpi_utils import Gatherv_rows
from pyuoi.lbfgs import fmin_lbfgs

class CV_Lasso():

    @classmethod
    def run(self, X, y, args):

        if 'comm' in list(args.keys()):
            comm = args['comm']
        else:
            comm = None

        n_alphas = args['n_alphas']
        cv_splits = 5

        lasso = LassoCV(cv = cv_splits, n_alphas = n_alphas).fit(X, y.ravel())

        return lasso
        
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

        uoi = UoI_Lasso(
            normalize=False,
            n_boots_sel=int(args['n_boots_sel']),
            n_boots_est=int(args['n_boots_est']),
            estimation_score=args['est_score'],
            stability_selection = args['stability_selection'],
            manual_penalty = manual_penalty,
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
