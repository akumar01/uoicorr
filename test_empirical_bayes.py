# Debug empirical bayes by re-creating the simulations in George and Foster
import numpy as np 
from sklearn.linear_model import LinearRegression
from scipy.special import xlogy
import pdb
import time

# Orthogonal design:
def orthogonal_sim():

    p = 1000
    n = 1000

    reps = 100
    c = 25
    q = [100, 200, 300, 400, 500]

    # Y = beta + epsilon
    lm = LinearRegression(fit_intercept=False)

    # Record losses
    CCML_losses = np.zeros((len(q), reps))
    BIC_losses = np.zeros((len(q), reps))


    for i1, q_ in enumerate(q):
        t0 = time.time() 
        for rep in range(reps):
            beta = np.zeros(p)
            beta[:q_] = np.random.normal(0, c, size = (q_,))

            y = beta + np.random.multivariate_normal(np.zeros(p), np.eye(p))

            ss_gamma = np.zeros(p + 1)
            beta_gamma = np.zeros((p + 1, p))

            # Calculate the sequence of ss_gamma values over all subsets of models
            for i3, gamma in enumerate(range(p + 1)):
                support = np.zeros(p)
                support[:gamma] = 1
                support = support.astype(bool)
                # p = n, so create a diagonal matrix
                X = np.diag(np.ones(p))
                X_gamma = X[:, support]

                beta_gamma[i3, support] = y[support]
                # Calculate ss_gamma
                ss_gamma[i3] = beta_gamma[i3, support].T @ X_gamma.T @ X_gamma @ beta_gamma[i3, support]


            # Evaluate BIC and empirical bayes on the model supports
            BIC = ss_gamma - np.log(n) * np.arange(p + 1)

            thres_log = lambda x: np.log(np.maximum(x, 1))
            CCML = ss_gamma - np.multiply(np.arange(p + 1), (1 + thres_log(np.divide(ss_gamma, np.arange(p + 1),
                                                             where=np.arange(p+1) != 0))))\
                            + 2 * (xlogy(p - np.arange(p + 1), p - np.arange(p + 1)) + xlogy(np.arange(p + 1), np.arange(p + 1)))

            BIC_idx = np.argmax(BIC)
            CCML_idx = np.argmax(CCML)

            # Evaluate losCMs
            BIC_losses[i1, rep] = (beta_gamma[BIC_idx, :] - y).T @ (beta_gamma[BIC_idx, :] - y)
            CCML_losses[i1, rep] = (beta_gamma[CCML_idx, :] - y).T @ (beta_gamma[CCML_idx, :] - y)

        print('q=%d, %f s' % (q_, time.time() - t0))
    
    return BIC_losses, CCML_losses


# Here, the point is not necessarily to compare to the paper, but to make sure that eBIC is giving
# reasonable results we can then check against
def nonorthog_sim():

    p = 500
    n = 1000

    reps = 100
    c = 25
    q = [100, 200, 300, 400, 500]

    # Y = beta + epsilon
    lm = LinearRegression(fit_intercept=False)

    # Record losses
    CCML_losses = np.zeros((len(q), reps))
    BIC_losses = np.zeros((len(q), reps))

    # Fix the design matrix
    X = np.random.multivariate_normal(mean = np.zeros(p), cov = np.eye(p), 
                                      size = n)

    for i1, q_ in enumerate(q):
        t0 = time.time() 
        for rep in range(reps):
            beta = np.zeros(p)
            beta[:q_] = np.random.normal(0, c, size = (q_,))

            y = X @ beta + np.random.multivariate_normal(np.zeros(n), np.eye(n))

            ss_gamma = np.zeros(p + 1)
            beta_gamma = np.zeros((p + 1, p))

            # Calculate the sequence of ss_gamma values over all subsets of models
            for i3, gamma in enumerate(range(p + 1)):
                support = np.zeros(p)
                support[:gamma] = 1
                support = support.astype(bool)
                X_gamma = X[:, support]

                if np.count_nonzero(1 * support) > 0:
                    beta_gamma[i3, support] = lm.fit(X_gamma, y).coef_
                # Calculate ss_gamma
                ss_gamma[i3] = beta_gamma[i3, support].T @ X_gamma.T @ X_gamma @ beta_gamma[i3, support]


            # Evaluate BIC and empirical bayes on the model supports
            BIC = ss_gamma - np.log(n) * np.arange(p + 1)

            thres_log = lambda x: np.log(np.maximum(x, 1))
            CCML = ss_gamma - np.multiply(np.arange(p + 1), (1 + thres_log(np.divide(ss_gamma, np.arange(p + 1),
                                                             where=np.arange(p+1) != 0))))\
                            + 2 * (xlogy(p - np.arange(p + 1), p - np.arange(p + 1)) + xlogy(np.arange(p + 1), np.arange(p + 1)))

            BIC_idx = np.argmax(BIC)
            CCML_idx = np.argmax(CCML)

            # Evaluate losCMs
            BIC_losses[i1, rep] = (X @ beta_gamma[BIC_idx, :] - y).T @ (X @ beta_gamma[BIC_idx, :] - y)
            CCML_losses[i1, rep] = (X @ beta_gamma[CCML_idx, :] - y).T @ (X @ beta_gamma[CCML_idx, :] - y)

        print('q=%d, %f s' % (q_, time.time() - t0))
    
    return BIC_losses, CCML_losses

