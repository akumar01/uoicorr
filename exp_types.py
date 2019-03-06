import numpy as np
import pdb
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from pyuoi.linear_model.lasso import UoI_Lasso
from pyuoi.linear_model.elasticnet import UoI_ElasticNet
from pyuoi.linear_model.gtv import GraphTotalVariance
import time

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

		uoi = UoI_Lasso(
			normalize=True,
			n_boots_sel=int(args['n_boots_sel']),
			n_boots_est=int(args['n_boots_est']),
			estimation_score=args['est_score'],
			stability_selection = args['stability_selection']
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

		uoi = UoI_ElasticNet(
			normalize=True,
			n_boots_sel=int(args['n_boots_sel']),
			n_boots_est=int(args['n_boots_est']),
			alphas = l1_ratios,
			estimation_score=args['est_score'],
			warm_start = False,
			stability_selection=args['stability_selection']
		)
		
		uoi.fit(X, y.ravel())
		return [uoi]	

class EN():

	@classmethod
	def run(self, X, y, args):

		l1_ratios = args['l1_ratios']
		n_alphas = args['n_alphas']
		cv_splits = 10

		if not isinstance(l1_ratios, np.ndarray):
			if np.isscalar(l1_ratios):
				l1_ratios = np.array([l1_ratios])
			else:
				l1_ratios = np.array(l1_ratios)


		alphas = np.zeros((l1_ratios.size, n_alphas))
		scores = np.zeros((l1_ratios.size, n_alphas))

		en = ElasticNet(normalize=True, warm_start = False)

		# Use 10 fold cross validation. Do this in a manual way to enable use of warm_start and custom parameter sweeps
		kfold = KFold(n_splits = cv_splits, shuffle = True)

		for l1_idx, l1_ratio in enumerate(l1_ratios):
			# Generate alphas to use
			alphas[l1_idx, :] = _alpha_grid(X = X, y = y.ravel(), l1_ratio = l1_ratio, normalize = True, n_alphas = n_alphas)

			for a_idx, alpha in enumerate(alphas[l1_idx, :]):

				en.set_params(alpha = alpha, l1_ratio = l1_ratio)

				cv_scores = np.zeros(cv_splits)
				# Cross validation splits into training and test sets
				for i, cv_idxs in enumerate(kfold.split(X, y)):
					en.fit(X[cv_idxs[0], :], y[cv_idxs[0]])
					cv_scores[i] = r2_score(y[cv_idxs[1]], en.coef_ @ X[cv_idxs[1], :].T)

				# Average together cross-validation scores
				scores[l1_idx, a_idx] = np.mean(cv_scores)

		# Select the model with the maximum score
		max_score_idx = np.argmax(scores.ravel())
		max_score_idxs = np.unravel_index(max_score_idx, (l1_ratios.size, n_alphas))
		en.set_params(l1_ratio = l1_ratios[max_score_idxs[0]], alpha = alphas[max_score_idxs[0], max_score_idxs[1]])
		en.fit(X, y.ravel())
		return [en]

class GTV():

	@classmethod
	def run(self, X, y, args):
		print('Finished imports')
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


		scores = np.zeros((lambda_S.size, lambda_TV.size, lambda_1.size))

		# Use k-fold cross_validation
		# kfold = KFold(n_splits = cv_splits, shuffle = True)
		models = []

		for i1, l1 in enumerate(lambda_S):
			for i2, l2 in enumerate(lambda_TV):
				for i3, l3 in enumerate(lambda_1):

					gtv = GraphTotalVariance(normalize=True, warm_start = False)
					gtv.set_params(lambda_S = l1, lambda_TV = l2, lambda_1 = l3)
					gtv.fit(X, y, cov)

					models.append(gtv)

		return models