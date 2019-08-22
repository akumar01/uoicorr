# Debug empirical bayes by re-creating the simulations in George and Foster
import numpy as np 
from sklearn.linear_model import LinearRegression
from scipy.special import xlogy

# Orthogonal design:
def orthogonal_sim():

    p = 1000
    n = 1000

    reps = 200
    c = 25
    q = [0, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000]

    # Y = beta + epsilon
    lm = LinearRegression(fit_intercept=False)

    # Record losses
    CCML_losses = np.zeros((len(q), reps))
    BIC_losses = np.zeros((len(q), reps))


    for i1, q_ in enumerate(q): 
        for rep in reps:
            beta = np.zeros(p)
            beta[:q_] = np.random.normal(0, c, size = (q_,))

            y = beta + np.random.multivariate_normal(np.zeros(p), np.eye(p))

            ss_gamma = np.zeros(p + 1)


            beta_gamma = np.zeros((p + 1, p + 1))

            # Calculate the sequence of ss_gamma values over all subsets of models
            for i3, gamma in enumerate(range(p + 1)):
                support = np.zeros(p)
                support[:gamma] = 1

                # p = n, so create a diagonal matrix
                X = np.diag(np.ones(p))
                X_gamma = X[:, support]

                beta_gamma[i3, support] = y[support]

                # Calculate ss_gamma
                ss_gamma[i3] = beta_gamma[i3, support].T @ X_gamma.T @ X_gamma @ beta_gamma[i3, support]

            # Evaluate BIC and empirical bayes on the model supports
            BIC = ss_gamma - np.log(n) * np.arange(p + 1)

            thres_log = lambda x: np.log(np.maximum(x, 1))
            CCML = ss_gamma - np.multiply(np.arange(p + 1), (1 + thres_log(np.divide(ss_gamma, np.arange(p + 1)))))\
                            + 2 * (xlogy(p - np.arange(p + 1), p - np.arange(p + 1)) + xlogy(np.arange(p + 1), np.arange(p + 1)))

            BIC_idx = np.argmax(BIC)
            CCML_idx = np.argmax(CCML)

            # Evaluate loss
            BIC_losses[i1, rep] = (beta_gamma[BIC_idx, :] - y).T @ (beta_gamma[BIC_idx, :] - y)
            CCML_losses[i1, rep] = (beta_gamma[CCML_idx, :] - y).T @ (beta_gamma[CCML_idx, :] - y)

    return BIC_losses, CCML_losses