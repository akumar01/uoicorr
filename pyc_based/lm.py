import pycasso

# Pycasso solver wrapper with minimal class structure to interface with UoI
class PycassoLasso():

    def __init__(self, fit_intercept = False, max_iter = 1000):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def init_solver(self, X, y, alphas):

        self.solver = pycasso.Solver(X, y, family = 'gaussian', 
                      useintercept = self.fit_intercept, lambdas = alphas,
                      penalty = 'l1', max_ite = self.max_iter)

    def fit(self, X, y, alphas):

        self.init_solver(X, y, alphas)
        self.solver.train()
        # Coefs across the entire solution path
        self.coef_ = self.solver.result['beta']
