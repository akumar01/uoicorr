import numpy as np
import time
import pdb

import pycasso
from sklearn.linear_model import LassoLars, lasso_path, LinearRegression
from scipy.special import binom

from info_criteria import mBIC, GIC, bayesian_lambda_selection

from sklearn.model_selection import train_test_split

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


# Fit a model using adaptive BIC criteria given sparsity estimates
def adaptive_BIC_estimator(X, y, sparsity_estimates, n_boots = 48, train_frac = 0.75):
        
    n_samples, n_features = X.shape
    
    np1 = 100
    penalties = np.linspace(0, 5 * np.log(n_samples), np1)
    
    oracle_penalty = np.zeros(n_boots)
    bayesian_penalty = np.zeros(n_boots)

    estimates = np.zeros((n_boots, n_features))
    oracle_estimates = np.zeros((n_boots, n_features))

    
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

        supports = (coefs.T != 0).astype(bool)

        # Stick the true model in there
        # supports = np.vstack([supports, (beta.ravel() !=0).astype(bool)])

        sa = selection_accuracy(beta.ravel(), supports)
        # Keep track of oracle performance
        MIC_scores_ = np.zeros((supports.shape[0], np1))

        boot_estimates = np.zeros((supports.shape[0], n_features))

        models = []
        
        for j in range(supports.shape[0]):
            
            support = supports[j, :]

            if np.count_nonzero(1 * support > 0):
                model = LinearRegression().fit(X[:, support] , y)
                boot_estimates[j, support] = model.coef_.ravel()
                y_pred = model.predict(X[:, support])
                models.append(model)
            else:
                y_pred = np.zeros(y.size)
                models.append(np.nan)
                
            MIC_scores_[j, :] =  np.array([MIC(y.ravel(), y_pred.ravel(), 
                                           np.count_nonzero(1 * support), penalty) 
                                           for penalty in penalties])

        selected_models = np.argmin(MIC_scores_, axis = 0)
        
        bayes_factors = np.zeros(np1)

        for i3, penalty in enumerate(penalties):

            support = supports[selected_models[i3], :]
            yy = models[selected_models[i3]].predict(X[:, support])

            ll_, p1_, BIC_, BIC2_, BIC3_, M_k_, P_M_ = bayesian_lambda_selection(
                                                       y, yy, n_features, 
                                                       np.count_nonzero(1 * support),
                                                       sparsity_estimates, penalty)

            # Add things up appropriately
            bayes_factors[i3] = 2 * ll_ - BIC_ - BIC2_ + BIC3_ - p1_ + P_M_


        # Select the penalty based on the highest bayes factors 
        bidx = np.argmax(bayes_factors)

        # Save the penalty strength and the chosen model
        bayesian_penalty[boot] = penalties[bidx]

        # For MIC scores, record the oracle selection accuracy and the oracle penalty
        MIC_selection_accuracies = [selection_accuracy(beta.ravel(), 
                                    supports[selected_models[j], :]) 
                                    for j in range(selected_models.size)]

        # For MIC scores, record the oracle selection accuracy and the orcale penalty 

        oracle_penalty[boot] = penalties[np.argmax(MIC_selection_accuracies)]

        #        MIC_oracle_sa_[boot] = np.max(MIC_selection_accuracies)    
#        bMIC_sa_[boot] = MIC_selection_accuracies[bidx]
        
        # Record the best estimate on this bootstrap as determined by the oracle
        # and by our procedure
        estimates[boot, :] = boot_estimates[selected_models[bidx], :]
        oracle_estimates[boot, :] = \
        boot_estimates[selected_models[np.argmax(MIC_selection_accuracies)], :]

    # Take the median and record final selection accuracies
    final_bayes_estimates = np.median(estimates, axis = 0)
    final_oracle_estimates = np.median(oracle_estimates, axis = 0)

    return final_bayes_estimates, final_oracle_estimates, oracle_penalty, bayesian_penalty, estimates
