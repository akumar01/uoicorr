import numpy as np
import time
import pdb
import pycasso

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import selection_accuracy
from info_criteria import mBIC, GIC, full_bayes_factor

# Sparsity estimation on the basis of BIC
def sparsity_estimator0(X, y, n_boots = 48, train_frac = 0.75):
    
    sparsity_estimates = np.zeros(n_boots)
    
    n_features, n_samples = X.shape
    
    for boot in range(n_boots):
        # Draw bootstraps
        idxs_train, idxs_test = train_test_split(np.arange(X.shape[0]), 
                                                 train_size = train_frac,
                                                 test_size = 1 - train_frac)
        Xb = X[idxs_train]
        yb = y[idxs_train]

        Xb = StandardScaler().fit_transform(Xb)
        yb -= np.mean(yb)

        # Use the pycasso solver
        solver = pycasso.Solver(Xb, yb, penalty='l1')        
        solver.train()

        coefs = solver.result['beta']
        y_pred = Xb @ coefs.T

        # Assess BIC on the LARS path to estimate the sparsity. 
        BIC_scores = np.array([GIC(yb.ravel(), y_pred[:, j].ravel(),
                                   np.count_nonzero(coefs[j, :]), np.log(n_samples)) 
                               for j in range(coefs.shape[0])])  
        sparsity_estimates[boot] = float(np.count_nonzero(
                            coefs[np.argmin(BIC_scores), :]))/float(n_features)

    return sparsity_estimates

# Refine sparsity estimate using mBIC
def sparsity_estimator1(X, y, s0, n_boots = 48, train_frac = 0.75):
        

    sparsity_estimates = np.zeros(n_boots)
    
    for boot in range(n_boots):
        # Draw bootstraps
        idxs_train, idxs_test = train_test_split(np.arange(X.shape[0]), 
                                                 train_size = train_frac,
                                                 test_size = 1 - train_frac)
        Xb = X[idxs_train]
        yb = y[idxs_train]

        Xb = StandardScaler().fit_transform(Xb)
        yb -= np.mean(yb)

        n_samples, n_features = Xb.shape

        # Use fast pycasso solver
        solver = pycasso.Solver(Xb, yb, penalty='l1')        
        solver.train()

        coefs = solver.result['beta']

        y_pred = Xb @ coefs.T
        
        mBIC_scores = np.zeros(coefs.shape[0])
        
        for j in range(coefs.shape[0]): 
        
            ll_, p1_, BIC_, BIC2_, BIC3_, M_k_, P_M_ = full_bayes_factor(yb, y_pred[:, j], n_features, 
                                                                                 np.count_nonzero(coefs[j, :]),
                                                                                 s0, 0)
            mBIC_scores[j] = 2 * ll_ - BIC_ - BIC2_ + BIC3_ + P_M_ 
            
        sparsity_estimates[boot] = float(np.count_nonzero(
                            coefs[np.argmax(mBIC_scores), :]))/float(n_features)

    return sparsity_estimates

# adaptive information criteria
def aBIC(X, y, estimates, true_model):

    n_boots = 48
    train_frac = 0.75

    n_samples, n_features = X.shape
    
    np1 = 100
    penalties = np.linspace(0, 5 * np.log(n_samples), np1)
    
    oracle_penalty = np.zeros(n_boots)
    bayesian_penalty = np.zeros(n_boots)

    # Also record progress of sparsity estimates
    sparsity_estimates = np.zeros((3, n_boots))

    # Step (1): Initial sparsity estimate
    sparsity_estimates_ = sparsity_estimator0(X, y)
    sparsity_estimates[0, :] = sparsity_estimates_

    # Step (2): Refine sparsity estimates
    sparsity_estimates_ = sparsity_estimator1(X, y, sparsity_estimates_)

    sparsity_estimates[1, :] = sparsity_estimates_

    # Step (3) : Throw it into the full bayes factor and obtain
    # oracle penalty, bayesian penalty, oracle selection index, bayesian selection index 
    oracle_penalty, bayesian_penalty, bidx, oidx, goft, pt = \
    bayesian_penalty_selection(X, y, estimates, sparsity_estimates, penalties, true_model)

    return oracle_penalty, bayesian_penalty, bidx, oidx, sparsity_estimates, goft, pt

# Follow a similar procedure as aBIC, but do not use an L0 penalty in the bayes factor
def mBIC(X, y, estimates):

    n_boots = 48
    train_frac = 0.75

    n_samples, n_features = X.shape

    sparsity_estimates = np.zeros((2, n_boots))

    # Step (1): Initial sparsity estimate
    sparsity_estimates_ = sparsity_estimator0(X, y)
    sparsity_estimates[0, :] = sparsity_estimates_

    # Step (2): Refine sparsity estimates
    sparsity_estimates_ = sparsity_estimator1(X, y, sparsity_estimates_)

    sparsity_estimates[1, :] = sparsity_estimates_

    # Step (3) : Calculate Bayes factors with no L0 penalty
    y_pred = np.array([X @ estimates[i, :] for i in range(estimates.shape[0])])

    bayes_factors = np.zeros(estimates.shape[0])

    for i in range(estimates.shape[0]):
        support = estimates[i, :] != 0

        ll_, p1_, BIC_, BIC2_, BIC3_, M_k_, P_M_ = full_bayes_factor(
                                                   y, y_pred[i, :], n_features, 
                                                   np.count_nonzero(1 * support),
                                                   sparsity_estimates_, 0)        
        # Add things up appropriately
        bayes_factors[i] = 2 * ll_ - BIC_ - BIC2_ + BIC3_ - p1_ + P_M_

    return bayes_factors        

# Fit a model using adaptive BIC criteria given sparsity estimates, no L0 prior nonesense


# Fit a model using adaptive BIC criteria given sparsity estimates
def bayesian_penalty_selection(X, y, estimates, sparsity_estimates, 
                               penalties, true_model):
    
    n_samples, n_features = X.shape

    y_pred = np.array([X @ estimates[i, :] for i in range(estimates.shape[0])])

    GIC_scores_ =  np.array([[GIC(y.ravel(), y_pred[i, :], 
                             np.count_nonzero(estimates[i, :]), penalty) 
                             for penalty in penalties]
                             for i in range(estimates.shape[0])])

    selected_models = np.argmin(GIC_scores_, axis = 0)

    bayes_factors = np.zeros(penalties.size)
    gof_terms = np.zeros(penalties.size)
    penalty_terms = np.zeros(penalties.size)

    for i, penalty in enumerate(penalties):

        support = estimates[selected_models[i], :] != 0

        yy = X @ estimates[selected_models[i], :]
        # By passing in 0, we exclude the L0 term from this bayes factor
        ll_, p1_, BIC_, BIC2_, BIC3_, M_k_, P_M_ = full_bayes_factor(
                                                   y, yy, n_features, 
                                                   np.count_nonzero(1 * support),
                                                   sparsity_estimates, 0)

        # Add things up appropriately
        bayes_factors[i] = 2 * ll_ - BIC_ - BIC2_ + BIC3_ - p1_ + P_M_
        gof_terms[i] = 2 * ll_

    # Save the penalty strength and the chosen model
    bayesian_penalty = penalties[np.argmax(bayes_factors)]
    gof_term = gof_terms[np.argmax(bayes_factors)]
    bidx = selected_models[np.argmax(bayes_factors)]
    penalty_term = bayesian_penalty * np.count_nonzero(estimates[bidx, :])

    # For MIC scores, record the oracle selection accuracy and the oracle penalty
    MIC_selection_accuracies = [selection_accuracy(true_model, 
                                estimates[selected_models[j], :]) 
                                for j in range(selected_models.size)]

    # For MIC scores, record the oracle selection accuracy and the orcale penalty  
    oracle_penalty = penalties[np.argmax(MIC_selection_accuracies)]
    oidx = selected_models[np.argmax(MIC_selection_accuracies)]
    

    return oracle_penalty, bayesian_penalty, bidx, oidx, gof_term, penalty_term
