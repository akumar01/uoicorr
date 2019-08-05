import numpy as np
import time
import pdb
import pycasso

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

        alphas, _, coefs  = lars_path(Xb, yb.ravel(), method = 'lasso')

        y_pred = Xb @ coefs

        # Assess BIC on the LARS path to estimate the sparsity. 
        BIC_scores = np.array([GIC(yb.ravel(), y_pred[:, j].ravel(),
                                   np.count_nonzero(coefs[:, j]), np.log(n_samples)) 
                               for j in range(coefs.shape[1])])  
        
        sparsity_estimates[boot] = float(np.count_nonzero(
                            coefs[:, np.argmin(BIC_scores)]))/float(n_features)

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

        # Use fast pycasso solver
        solver = pycasso.Solver

        alphas, _, coefs  = lars_path(Xb, yb.ravel(), method = 'lasso')

        y_pred = Xb @ coefs
        
        mBIC_scores = np.zeros(coefs.shape[1])
        
        for j in range(coefs.shape[1]): 
        
            ll_, p1_, BIC_, BIC2_, BIC3_, M_k_, P_M_ = bayesian_lambda_selection(yb, y_pred[:, j], n_features, 
                                                                                 np.count_nonzero(coefs[:, j]),
                                                                                 s0, 0)
            mBIC_scores[j] = 2 * ll_ - BIC_ - BIC2_ + BIC3_ + P_M_ 
            
        sparsity_estimates[boot] = float(np.count_nonzero(
                            coefs[:, np.argmax(mBIC_scores)]))/float(n_features)

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
    sparsity_estimates_ = sparsity_estimator1(X, y)

    sparsity_estimates[1, :] = sparsity_estimates_

    # Step (3) : Throw it into the full bayes factor and obtain
    # oracle penalty, bayesian penalty, oracle selection index, bayesian selection index 
    oracle_penalty, bayesian_penalty, bidx, oidx = \
    bayesian_penalty_selection(X, y, estimates, sparsity_estimates, penalties, true_model)

    return oracle_penalty, bayesian_penalty, bidx, oidx

# Fit a model using adaptive BIC criteria given sparsity estimates
def bayesian_penalty_selection(X, y, estimates, sparsity_estimates, 
                               penalties, true_model):
                        
    GIC_scores_ =  np.array([GIC(y.ravel(), y_pred.ravel(), 
                                 np.count_nonzero(estimates[i, :]), penalty) 
                             for i in range(estimates.shape[0])
                             for penalty in penalties])

    selected_models = np.argmin(GIC_scores_, axis = 0)
        
    bayes_factors = np.zeros(penalties.size)

    for i, penalty in enumerate(penalties):

        support = estimates[selected_models[i], :] != 0
        yy = models[selected_models[i]].predict(X[:, support])

        ll_, p1_, BIC_, BIC2_, BIC3_, M_k_, P_M_ = full_bayes_factor(
                                                   y, yy, n_features, 
                                                   np.count_nonzero(1 * support),
                                                   sparsity_estimates, penalty)

        # Add things up appropriately
        bayes_factors[i] = 2 * ll_ - BIC_ - BIC2_ + BIC3_ - p1_ + P_M_


    # Select the penalty based on the highest bayes factors 
    bidx = np.argmax(bayes_factors)

    # Save the penalty strength and the chosen model
    bayesian_penalty = penalties[bidx]

    # For MIC scores, record the oracle selection accuracy and the oracle penalty
    MIC_selection_accuracies = [selection_accuracy(beta.ravel(), 
                                estimates[selected_models[j], :]) 
                                for j in range(selected_models.size)]

    # For MIC scores, record the oracle selection accuracy and the orcale penalty 

    oracle_penalty = penalties[np.argmax(MIC_selection_accuracies)]
    oidx = selected_models[np.argmax(MIC_selection_accuracies)]

    return oracle_penalty, bayesian_penalty, bidx, oidx
