import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import pdb

# R package wrapped in sklearn-like interface
# Note: there is no way to set fit_intercept = False, so make sure the data 
# are centered

class SLOPE():
	''' class SLOPE : sorted L1 ordered penalized estimator. Borrowed from R
			lambda_method : 'FDR', 'user'
				the method to use to generate the list of regularization parameters
			lambda_spec : 
				argument that corresponds to the chosen lambda_method. For FDR, this is 
				a (scalar) desired false discovery rate. 
				For user, this is a list of lambdas with 
				the same length as the number of features'''

	def __init__(self, lambda_method='FDR', lambda_spec=0.2):

		# Import R package
		self.slope = importr('SLOPE')
		numpy2ri.activate()

		if lambda_method not in ['FDR', 'user']:
			raise ValueError('lambda method not recognized')

		self.lambda_method = lambda_method

		self.lambda_spec = lambda_spec

	def get_lambda(self, X, y):

		# Obtain lambdas
		n_samples, n_features = X.shape

		if self.lambda_method == 'FDR':
			lambdas = self.slope.create_lambda(n_samples, n_features, 
											   fdr = self.lambda_spec,
										  	   method='gaussian')
		elif self.lambda_method == 'user':
			lambdas = self.lambda_spec['lambda']		

		return lambdas

	def fit(self, X, y):

		self.lambdas = self.get_lambda(X, y)
		if np.ndim(y) == 1:
			y = y[:, np.newaxis]
		result = self.slope.SLOPE_solver(X, y, self.lambdas)
		# Is the intercept present?
		self.coef_ = np.array(result.rx2('x'))

class SLOPE_CV(SLOPE):
	''' class SLOPE_CV : cross-validator for SLOPE 
		
		n_folds : Number of CV folds to do
		
		lambda_method : 'FDR', 'user'
			the method to use to generate the list of regularization parameters
		
		lambda_spec : *args
			arguments that correspond to the chosen lambda_method. For FDR, this is 
			a list of false discovery rates. For user, this an n_lambda_sequences x n_features
			ndarray
	'''

	def __init__(self, nfolds = 5, lambda_method = 'FDR', lambda_spec=[0.2]):

		self.nfolds = 5

		super(SLOPE_CV, self).__init__(lambda_method, lambda_spec)

	def get_lambda(self, X, y):
		# Obtain lambda sequences

		n_samples, n_features = X.shape

		if self.lambda_method == 'FDR':

			lambdas = np.array([self.slope.create_lambda(n_samples, n_features, fdr) 
								for fdr in self.lambda_spec])
		elif self.lambda_method == 'user':

			lambdas = self.lambda_spec

		return lambdas


	def fit(self, X, y):

		# Obtain lambdas from the full dataset

		lambdas = self.get_lambda(X, y)

		cross_validator = KFold(n_splits = self.nfolds)

		# Store scores
		scores = np.zeros((lambdas.shape[0], self.nfolds))

		if np.ndim(y) == 1:
			y = y[:, np.newaxis]

		for i in range(lambdas.shape[0]):

			fold_idx = 0

			for train_idxs, test_idxs in cross_validator.split(X):
				X_train = X[train_idxs]
				y_train = y[train_idxs]

				X_test = X[test_idxs]
				y_test = y[test_idxs]
				result = self.slope.SLOPE_solver(X_train, y_train, lambdas[i, :])
				coefs = np.array(result.rx2('x'))

				scores[i, fold_idx] = r2_score(y_test, X_test @ coefs)

				fold_idx += 1

		# Average over folds
		scores = np.mean(scores, axis = 1)

		self.scores = scores

		best_idx = np.argmax(scores)

		self.lambda_ = lambdas[best_idx, :]

		if self.lambda_method == 'FDR':

			self.fdr_ = self.lambda_spec[best_idx]

		# refit with the best performing lambda
		final_result = self.slope.SLOPE_solver(X, y, lambdas[best_idx, :])

		self.coef_ = np.array(result.rx2('x'))


