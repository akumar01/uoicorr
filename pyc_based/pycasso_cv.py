import numpy as np
import pycasso
from sklearn.model_selection import KFold
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from . import lm
import pdb

# Wrapper class to stitch together multiple path-wise solutions from 
# pycasso corresponding to different gamma 
class PycassoGrid():
	'''	class PycassoGrid : Fit the pycasso solver pathwise on a grid of regularization 
		parameters.
		
		penalty : 'l1' 'scad' 'mcp' (for l1, can just use Pycassolasso if desired)
	
	'''

	def __init__(self, penalty, n_alphas=100, gamma = [3], fit_intercept = False, 
				 eps = 1e-3, alphas=None):

		self.penalty = penalty
		self.n_alphas = n_alphas
		if np.isscalar(gamma):
			gamma = [gamma]
		self.gamma = np.array(gamma)
		self.fit_intercept = fit_intercept
		self.eps = eps
		self.alphas = alphas

	def fit(self, X, y):

		_, n_features = X.shape

		if self.alphas is None:
			self.alphas = self.get_alphas(X, y)
		else:
			self.n_alphas = self.alphas.size

		coefs = np.zeros((self.gamma.size, self.n_alphas, n_features))

		for gidx, gamma in enumerate(self.gamma):

			solver = pycasso.Solver(X, y, family = 'gaussian', 
									penalty = self.penalty, gamma = gamma,
									useintercept = self.fit_intercept, 
									lambdas = self.alphas)
			solver.train()
			coefs[gidx, ...] = solver.result['beta']

		self.coef_ = coefs

	def get_alphas(self, X, y):

		# The lambda selection that pycasso uses is essentially the same
		# as alpha_grid
		return _alpha_grid(X, y, n_alphas = self.n_alphas, eps = self.eps)

	# Predict over the entire grid
	def predict_grid(self, X):

		y_pred = self.coef_ @ X.T
		return y_pred

	# Score over the whole grid
	def score_grid(self, X, y):

		scores = np.zeros((self.gamma.size, self.n_alphas))

		y_pred = self.predict_grid(X)

		for gidx in range(self.gamma.size):

			scores[gidx, :] = np.array([r2_score(y, y_pred[gidx, j, :])
							  for j in range(self.n_alphas)])

		return scores

# Wrapper class to automate cross-validation with pycasso solvers
# penalty: 'l1' , 'mcp', or 'scad' 
class PycassoCV(PycassoGrid):

	def __init__(self, penalty, n_alphas=100, gamma = [3], 
				 nfolds = 5, fit_intercept = False, eps = 1e-3, 
				 alphas=None):

		self.nfolds = nfolds
		super(PycassoCV, self).__init__(penalty, n_alphas, gamma, 
										fit_intercept, eps, alphas)


	def fit(self, X, y):

		# The lambda selection that pycasso uses is essentially the same
		# as alpha_grid
		if self.alphas is None:
			self.alphas = _alpha_grid(X, y, n_alphas = self.n_alphas, eps = self.eps)
		else:
			self.n_alphas = self.alphas.size
		# Initialize cross-validator object
		self.cross_validator = KFold(n_splits = self.nfolds)

		# Store scores
		scores = np.zeros((self.nfolds, self.gamma.size, self.n_alphas))

		fold_idx = 0
		for train_idxs, test_idxs in self.cross_validator.split(X):
			X_train = X[train_idxs]
			y_train = y[train_idxs]

			X_test = X[test_idxs]
			y_test = y[test_idxs]

			super(PycassoCV, self).fit(X_train, y_train)

			scores[fold_idx, ...] = self.score_grid(X_test, y_test)

			fold_idx += 1

		# Average over folds
		scores = np.mean(scores ,axis = 0)

		self.scores = scores

		best_idx = np.unravel_index(np.argmax(scores.ravel()), scores.shape)

		# Set the selected parameters
		self.gamma_ = self.gamma[best_idx[0]]
		self.alpha_ = self.alphas[best_idx[1]]

		# Dummy regularization path (throw away all but the first)
		dummy_alphas = np.array([self.alpha_, self.alpha_/2, self.alpha_/4])

		# Refit with the selected parameters
		solver = pycasso.Solver(X, y, family = 'gaussian',
								penalty = self.penalty, gamma = self.gamma_, 
								useintercept = self.fit_intercept, 
								lambdas = dummy_alphas)

		solver.train()
		# Store final coefficients
		self.coef_ = solver.result['beta'][0, :]

# Cross-validator for PycassoElasticNet
class PycEnCV(lm.PycassoElasticNet):

	def __init__(self, n_folds=5, fit_intercept=False, max_iter=1000, lambda1 = None,
				 lambda2 = None):

		self.nfolds = n_folds
		super(PycEnCV, self).__init__(fit_intercept, max_iter, lambda1, lambda2)

	def fit(self, X, y):

		self.init_reg_params(X, y)
		if np.isscalar(self.lambda2):
			self.lambda2 = np.array([self.lambda2])

		cross_validator = KFold(n_splits = self.nfolds)

		scores = np.zeros((self.lambda2.size, self.lambda1.size, 
						   self.nfolds))

		# Outer loop over lambda2
		for i, l2 in enumerate(self.lambda2):

			fold_idx = 0

			for train_idxs, test_idxs in cross_validator.split(X):
				X_train = X[train_idxs]
				y_train = y[train_idxs]

				X_test = X[test_idxs]
				y_test = y[test_idxs]

				en = lm.PycassoElasticNet(fit_intercept=self.fit_intercept, max_iter=self.max_iter,
									   lambda1 = self.lambda1, lambda2 = l2)
				en.fit(X_train, y_train)

				y_pred = en.coef_ @ X_test.T

				scores[i, :, fold_idx] = np.array([r2_score(y_test, y_pred[j, :]) for j in
												   range(self.lambda1.size)])
				fold_idx += 1

		# Average over folds
		scores = np.mean(scores, axis = -1)
		self.scores = scores
		best_idx = np.unravel_index(np.argmax(scores), (self.lambda2.size, self.lambda1.size))

		self.lambda2_ = self.lambda2[best_idx[0]]
		self.lambda1_ = self.lambda1[best_idx[1]]

		# Refit with the selected parameters. 
		en = lm.PycassoElasticNet(fit_intercept = self.fit_intercept, max_iter=self.max_iter,
							   lambda1=self.lambda1_, lambda2=self.lambda2_)
		en.fit(X, y)

		self.coef_ = en.coef_

