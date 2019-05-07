import numpy as np
from numpy.linalg import norm, svd, pinv
import pdb
from scipy.optimize import minimize
import quadprog
# import cvxopt
import networkx as nx
from networkx import minimum_spanning_tree

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.linear_model.base import _pre_fit
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.utils import check_array, check_X_y
# from .base import AbstractUoILinearRegressor

# Importing from pip-installed PyUoI
from pyuoi.lbfgs import fmin_lbfgs

# Use pytorch to calculate numerical derivatives
import torch

# class UoI_GTV(AbstractUoILinearRegressor):

#     def __init__(self, lambda_1 = 1, lambda_TV = 1, lambda_S = 1,
#                  alphas=np.array([0.5]), n_boots_sel=48, 
#                  n_boots_est=48, selection_frac=0.9,
#                  estimation_frac=0.9, stability_selection=1.,
#                  estimation_score='r2', warm_start=True, eps=1e-3,
#                  copy_X=True, fit_intercept=True, normalize=True,
#                  random_state=None, max_iter=1000,
#                  comm=None):
#         super(UoI_GTV, self).__init__(
#             n_boots_sel=n_boots_sel,
#             n_boots_est=n_boots_est,
#             selection_frac=selection_frac,
#             estimation_frac=estimation_frac,
#             stability_selection=stability_selection,
#             estimation_score=estimation_score,
#             copy_X=copy_X,
#             fit_intercept=fit_intercept,
#             normalize=normalize,
#             random_state=random_state,
#             comm=comm
#         )

#         self.warm_start = warm_start
#         self.eps = eps
#         self.lambda_S = lambda_S
#         self.lambda_TV = lambda_TV
#         self.lambda_1 = lambda_1
#         self.__selection_lm = GraphTotalVariance(
#             lambda_S = lambda_S,
#             lambda_TV = lambda_TV,
#             lambda_1 = lambda_1,
#             max_iter=max_iter,
#             copy_X=copy_X,
#             warm_start=warm_start,
#             random_state=random_state)
#         self.__estimation_lm = LinearRegression()

#     @property
#     def estimation_lm(self):
#         return self.__estimation_lm

#     @property
#     def selection_lm(self):
#         return self.__selection_lm

#     def get_reg_params(self, X, y):
#         """Calculates the regularization parameters (alpha and lambda) to be
#         used for the provided data.

#         Note that the Elastic Net penalty is given by

#                 1 / (2 * n_samples) * ||y - Xb||^2_2
#             + lambda * (alpha * |b|_1 + 0.5 * (1 - alpha) * |b|^2_2)

#         where lambda and alpha are regularization parameters.

#         Note that scikit-learn does not use these names. Instead, scitkit-learn
#         denotes alpha by 'l1_ratio' and lambda by 'alpha'.

#         Parameters
#         ----------
#         X : array-like, shape (n_samples, n_features)
#             The design matrix.

#         y : array-like, shape (n_samples)
#             The response vector.

#         Returns
#         -------
#         reg_params : a list of dictionaries
#             A list containing dictionaries with the value of each
#             (lambda, alpha) describing the type of regularization to impose.
#             The keys adhere to scikit-learn's terminology (lambda->alpha,
#             alpha->l1_ratio). This allows easy passing into the ElasticNet
#             object.
#         """

#         # place the regularization parameters into a list of dictionaries
#         reg_params = list()
#         for l1_idx, l1 in enumerate(self.lambda_1):
#             for l2_idx, l2 in enumerate(self.lambda_S):
#                 for l3_idx, l3 in enumerate(self.lambda_TV):
#                     # reset the regularization parameter
#                     reg_params.append(dict(lambda_1=l1, 
#                                            lambda_S=l2,
#                                            lambda_TV = l3))

#         return reg_params


class GraphTotalVariance(ElasticNet):

    # use_skeleton: Whether to use the minimum spanning tree instead of 
    # the full covariance matrix
    # threshold: Threshold small values of the covariance matrix
    # minimizer: Choice of minimizer 

    def __init__(self, lambda_S, lambda_TV, lambda_1, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic', use_skeleton = False,
                 threshold = False, minimizer = 'quadprog'):

        super(GraphTotalVariance, self).__init__(
        fit_intercept=fit_intercept,
        normalize=normalize, precompute=precompute, copy_X=copy_X,
        tol=tol, warm_start=warm_start, positive=positive,
        random_state=random_state, selection=selection)

        self.lambda_S = lambda_S
        self.lambda_TV = lambda_TV
        self.lambda_1 = lambda_1

        self.use_skeleton = use_skeleton
        self.threshold = threshold
        self.minimizer = minimizer

    # # Find the maximum spanning graph of the covariance matrix to speed up
    # # computation using Prim's algorithm.
    # def skeleton_graph(self, sigma):

    #     p = sigma.shape[0]
        
    #     # Remove diagonal elements from sigma:
    #     sigma = sigma - np.diag(np.diag(sigma))

    #     # Following the terminology of the wikipedia page for Prim

    #     # Deviate slightly from wikipedia here - what we ultimately want to 
    #     # output is an adjacency matrix:
    #     MST = np.identity(p)

    #     # Set of vertices not yet associated with the MST
    #     Q = np.arange(p)

    #     # Candidate edge set
    #     E = []
    #     # Weights associated with candidate edge set
    #     C = []

    #     # Choose a vertex at random 
    #     v = np.random.choice(Q)
    #     Q = np.delete(Q, v)

    #     while Q.size > 0:

    #         # Add edge set of v to E. Edges are tuples. Ignore edges 
    #         # that lead to vertices that have already been visited
    #         edge_indices = np.arange(p)[sigma[v, :] != 0]

    #         E.extend([(v, w) for w in edge_indices if w in Q])

    #         # Add the associated weights to C
    #         C.extend([sigma[v, w] for w in edge_indices if w in Q])

    #         # Now select the edge in E that has the maximum edge weight
    #         # associated with it

    #         max_val = np.max(np.array(C))
    #         max_idx = np.argmax(np.array(C))
    #         try:
    #             MST[E[max_idx][0], E[max_idx][1]] = max_val
    #         except:
    #             pdb.set_trace()

    #         # Now, remove this edge from the list of edges and remove its edge weight
    #         # from the list of edge weights
    #         w = E[max_idx][1]

    #         del E[max_idx]
    #         del C[max_idx]
            
    #         # Remove the identified vertex from Q, and set v equal to it
    #         try:
    #             Q = np.delete(Q, np.where(Q == w))
    #         except:
    #             pdb.set_trace()
    #         v = w

    #     # Explicitly symmetrize:
    #     MST = MST + MST.T

    #     # Make the diagonal values half of what they are
    #     MST = MST - np.identity(p)

    #     return MST

    def skeleton_graph(self, sigma):

        # Convert to networkx graph
        G = nx.from_numpy_matrix(sigma)

        # Calculate MST
        MST = minimum_spanning_tree(G)

        return nx.convert_matrix.to_numpy_matrix(MST)

    # Transform the GTV objective into a quadratic programming problem
    # of the form 1/2 X^T Q X + a^T X subject to C X >= b where the first
    # meq constraints are equality constraints
    def gtv_quadprog(self, *args):


        #### Transform GTV into a generalized lasso ####

        # args: lambda_S, lambda_TV, lambda_1, X, y, cov
        lambda_S = args[0]
        lambda_TV = args[1]
        lambda_1 = args[2]
        X = args[3]
        y = args[4]
        cov = args[5]
        n = X.shape[0]
        p = X.shape[1]

        # Assemble edge set from the covariance matrix:
        E = []
        for i in range(p):
            for j in range(p):
                if i == j: 
                    continue
                if cov[i, j] != 0:
                    E.append([i, j])

        # Coordinate transformations:   
        Gamma = np.zeros((len(E), p))

        for i in range(Gamma.shape[0]):
            e_jl = np.zeros(p)
            e_kl = np.zeros(p)
            e_jl[E[i][0]] = 1
            e_kl[E[i][1]] = 1
            Gamma[i, :] = np.sqrt(cov[E[i][0], E[i][1]]) * (e_jl - np.sign(cov[E[i][0], E[i][1]]) * e_kl)

        XX = np.concatenate([X, np.sqrt(n * lambda_S) * Gamma])
        YY = np.concatenate([y, np.zeros((len(E), 1))])
        GG = np.concatenate([lambda_TV * Gamma, np.identity(p)])

        ### Transform generalized lasso into a constrained lasso ###

        # Singular value decomposition of GG:

        U, S, V = svd(GG)

        # GG will always have full column rank r = p. Divide into U_1 with dimension p and U_2 with dimension m - p
        r = S.size
        U1 = U[:, 0:r]
        U2 = U[:, r::]

        # Transform X
        XX = XX @ pinv(GG)        

        # Constraints

        # Equality constraints: U_2^T alpha = 0  (introduced by constrained lasso form)
        # Break up alpha into alpha_+ and alpha_-
        # Inequality constraints: All alpha_+ and alpha_- coefficients must be >= 0

        # Horizontal tiling
        C = np.concatenate([U2.T, -U2.T], axis = 1)
        # Combine equality and inequality constraints
        C = np.concatenate([C, np.identity(C.shape[1])])
        b = np.zeros(C.shape[0])

        # Quadratic programming objective function
        Q =  1/n * XX.T @ XX
        a =  1/n * XX.T @ YY

        # Enlarge the dimension of Q to handle the positive/negative decomposition
        Q = np.concatenate([Q, -Q], axis = 1)
        Q = np.concatenate([Q, -Q])

        a = lambda_1 * np.ones(Q.shape[0]) - np.concatenate([a, -a]).ravel()

        meq = U2.shape[1]

        # Need a symmetric, positive definite matrix for solvers 
        Q = 1/2 * (Q + Q.T + np.identity(Q.shape[0]) * 1e-6)

        # quadprog subtracts the linear term
        a = -a.ravel()

        # Transpose C
        C = C.T

        # Feed into quadprog
        solution = quadprog.solve_qp(Q, a, C, b, meq)

        # Recover actual coefficients

        coeffs_pm = solution[0]
        coeffs = coeffs_pm[0:int(len(coeffs_pm)/2)] - coeffs_pm[int(len(coeffs_pm)/2)::]

        # Invert the transformation on the betas
        betas = pinv(GG) @ coeffs

        return betas

    # Test to see whether we can make ordinary lasso work with quadratic programming
    def lasso_quadprog(self, *args):
        lambda1 = args[0]
        X = args[1]
        y = args[2]

#         n = X.shape[0]
#         p = X.shape[1]


# #        t = 1/lambda1         

#         # Constraints
#         # Inequality constraint matrix:
# #        A = np.concatenate([np.ones((1, p)) , -1* np.ones((1, p))], axis = 1)
# #        A = np.concatenate([A, -1*np.identity(2 * p)])

#         A = np.identity(2 * p)
    
#         # Inequality constraint vector:
# #        h = np.concatenate([np.array([t]), np.zeros(2*p)])

#         # Coefficients must be greater than 0
#         h = np.zeros(2 * p)

#         Q = 1/n * X.T @ X
#         c = 1/n * X.T @ y
#         t = p/lambda1         

#         # Constraints
#         # Inequality constraint matrix:
#         A = np.concatenate([np.ones((1, 2 * p)), -1*np.identity(2 * p)])
# #        A = np.concatenate([np.identity(p), np.identity(p)], axis = 1)
# #        A = np.concatenate([A, np.identity(2 * p)])

#         # Inequality constraint vector:
#         h = np.concatenate([np.array([t]), np.zeros(2*p)])
# #        h = np.concatenate([t*np.ones(p), np.zeros(2 * p)])
#         Q = 1/n*X.T @ X
#         c = 1/n*-X.T @ y

#         # Enlarge the dimension of Q to handle the positive/negative decomposition
#         QQ = np.concatenate([Q, -Q], axis = 1)
#         QQ = np.concatenate([QQ, -QQ])

#         cc = lambda1 * np.ones(2 * p) - np.concatenate([c, -c]).ravel()

#         return QQ, cc, A, h

    # Output scalar loss function and use pytorch to calculate its gradient
    def gtv_loss(self, beta, gradient, l1, ltv, ls, Xt, yt):

        n = Xt.shape[0]

        if beta.ndim == 1:
            beta = beta[:, np.newaxis]

        # Convert beta to pytorch tensor
        beta_t = torch.tensor(beta, requires_grad = True)

        loss = 1/n * torch.norm(yt - torch.mm(Xt, beta_t))**2
        #loss += l1 * ltv * torch.norm(torch.mm(Gamma_t, beta_t), 1)

        # loss = 1/n * torch.norm(y - torch.mm(X, beta))**2 + ls * torch.norm(torch.mm(Gamma, beta))**2\
        #         + l1 * ltv * torch.norm(torch.mm(Gamma, beta), 1)

        # Backpropagate gradient
        loss.backward()
        # Gradient of loss with respect to beta, making sure to detach from the pytorch graph
        dlossdbeta = beta_t.grad
        gradient[:] = dlossdbeta.detach().cpu().numpy().astype(float).ravel()

        return loss.detach().cpu().numpy().astype(float)

    def minimize(self, lambda_S, lambda_TV, lambda_1, X, y, cov):
        # use quadratic programming to optimize the GTV loss function

        if self.threshold:
            cov[cov < 0.01 * np.amax(cov)] = 0
        
        if self.use_skeleton:
            cov = self.skeleton_graph(cov)

        print(self.threshold)
        print(self.use_skeleton)
        print(np.count_nonzero(cov))

        if self.minimizer == 'quadprog':
           betas = self.gtv_quadprog(lambda_S, lambda_TV, lambda_1, X, y, cov)
        elif self.minimizer == 'lbfgs':
            # Use random initialization of beta weights

            n = X.shape[0]
            p = X.shape[1]

            # E: edge set of cov
            E = []
            for i in range(p):
                for j in range(p):
                    if i == j: 
                        continue
                    if cov[i, j] != 0:
                        E.append([i, j])

            # m: size of edge set
            m = len(E)

            # Gamma: Used to vectorize terms in the loss function involving the covariance matrix
            Gamma = np.zeros((m, p))

            for i in range(Gamma.shape[0]):
                e_jl = np.zeros(p)
                e_kl = np.zeros(p)
                e_jl[E[i][0]] = 1
                e_kl[E[i][1]] = 1
                Gamma[i, :] = np.sqrt(cov[E[i][0], E[i][1]]) * (e_jl - np.sign(cov[E[i][0], E[i][1]]) * e_kl)

            # Transform variables to write loss term compactly
            XX = np.concatenate([X, np.sqrt(n * lambda_S) * Gamma])
            YY = np.concatenate([y, np.zeros((len(E), 1))])
            GG = np.concatenate([lambda_TV * Gamma, np.identity(p)])

            XX = XX @ pinv(GG)        

            # Convert everything else to a torch tensor
            yt = torch.tensor(YY, requires_grad = False)
            Xt = torch.tensor(XX, requires_grad = False)

            betas = fmin_lbfgs(self.gtv_loss, np.zeros(m + p), 
                args = (lambda_1, lambda_TV, lambda_S, Xt, yt), orthantwise_c = lambda_1,
                max_linesearch = 40, epsilon = 1e-3, ftol = 1e-3)

            # Apply inverse transformation
            betas = pinv(GG) @ betas

        return betas

    # def cvx_minimize(self, lambda_S, lambda_TV, lambda_1, X, y, cov):
    #     Q, c, A, h = self.lasso_quadprog(lambda_1, X, y)

    #     # Put matrices in proprietary cvxopt format
    #     Q = 1/2 * (Q + Q.T)
    #     args = [cvxopt.matrix(Q), cvxopt.matrix(c), cvxopt.matrix(A), cvxopt.matrix(h)]
    #     sol = cvxopt.solvers.qp(*args)


    #     coeffs_pm = np.array(sol['x']).reshape((Q.shape[1],))
    #     coeffs = coeffs_pm[0:int(len(coeffs_pm)/2)] - coeffs_pm[int(len(coeffs_pm)/2)::]

    #     return coeffs


    def fit(self, X, y, cov):

        """Fit model with coordinate descent.
        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        cov : Estimated data covariance matrix

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.
        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        # Remember if X is copied
        X_copied = False
        X_copied = self.copy_X and self.fit_intercept
        X, y = check_X_y(X, y, accept_sparse='csc',
                         order='F', dtype=[np.float64, np.float32],
                         copy=X_copied, multi_output=True, y_numeric=True)
        y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                        ensure_2d=False)

        # Ensure copying happens only once, don't do it again if done above
        should_copy = self.copy_X and not X_copied
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=should_copy,
                     check_input=True)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        for k in range(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None

            coef_[k] = self.minimize(self.lambda_S, self.lambda_TV, self.lambda_1, X, y, cov)
#            coef_[k] = self.cvx_minimize(self.lambda_S, self.lambda_TV, self.lambda_1, X, y, cov)
        if n_targets == 1:
            self.coef_ = coef_[0]
        else:
            self.coef_ = coef_

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        # return self for chaining fit and predict calls
        return self
