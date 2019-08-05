import numpy as np
import pycasso
from sklearn.model_selection import KFold
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.metrics import r2_score

import pdb

# Wrapper class to stitch together multiple path-wise solutions from 
# pycasso corresponding to different gamma 
class PycassoGrid():

	def __init__(self, penalty, n_alphas=100, gamma = [3], fit_intercept = False, 
				 eps = 1e-3):

		self.penalty = penalty
		self.n_alphas = n_alphas
		self.gamma = np.array(gamma)
		self.fit_intercept = fit_intercept
		self.eps = eps

	def fit(self, X, y):

		_, n_features = X.shape

		if not hasattr(self, 'alphas'):
			self.alphas = self.get_alphas(X, y)

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
				 nfolds = 5, fit_intercept = False, eps = 1e-3):

		self.nfolds = nfolds
		super(PycassoCV, self).__init__(penalty, n_alphas, gamma, 
										fit_intercept, eps)


	def fit(self, X, y):

		# The lambda selection that pycasso uses is essentially the same
		# as alpha_grid
		self.alphas = _alpha_grid(X, y, n_alphas = self.n_alphas, eps = self.eps)

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

		# Ravel 
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
