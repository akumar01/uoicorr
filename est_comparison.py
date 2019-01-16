import time
import h5py
import pdb
import argparse, sys, os
import importlib
import numpy as np
from sklearn.metrics import r2_score

from pyuoi.linear_model.elasticnet import UoI_ElasticNet

from utils import gen_data, gen_beta, block_covariance

# Time execution
total_start = time.time()

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('--arg_file', default=None)

### Import arguments from file ###
try:
    arg_file_path, arg_file = os.path.split(parser.parse_args().arg_file)
    sys.path.append(arg_file_path)
    args = importlib.import_module(arg_file) 
except:
    print('Warning! Could not load arg file')

# Get parameters from file
correlation = args.correlations
block_size = args.block_size
n_features = args.n_features
sparsity = args.sparsity
n_samples = args.n_samples
n_features = args.n_features
reps = args.reps
results_file = args.results_file

if not isinstance(sparsity, np.ndarray):
    if np.isscalar(sparsity):
        sparsity = np.array([sparsity])
    else:
        sparsity = np.array(sparsity)

if not isinstance(n_samples, np.ndarray):
    if np.isscalar(n_samples):
        n_samples = np.array([n_samples])
    else:
        n_samples = np.array(n_samples)



# Record scores of model performance on test data, as well as false positive and false negatives
r2_scores = np.zeros((reps, sparsity.size, n_samples.size, len(block_size)))
BIC_scores = np.zeros(r2_scores.shape)
AIC_scores = np.zeros(r2_scores.shape)
AICc_scores = np.zeros(r2_scores.shape)
r2_fp = np.zeros(r2_scores.shape)
r2_fn = np.zeros(r2_scores.shape)
BIC_fp = np.zeros(r2_scores.shape)
BIC_fn = np.zeros(r2_scores.shape)
AIC_fp = np.zeros(r2_scores.shape)
AIC_fn = np.zeros(r2_scores.shape)
AICc_fp = np.zeros(r2_scores.shape)
AICc_fn = np.zeros(r2_scores.shape)

betas = np.zeros((sparsity.size, n_samples.size, len(block_size), n_features))

# Results file
results = h5py.File(results_file, 'w')

for ns_idx, ns in enumerate(n_samples):
    for sidx, s in enumerate(sparsity):
        for bidx, b in enumerate(block_size):
            # Generate consistent set of model parameters across repititions
            beta = gen_beta(n_features = n_features, sparsity = s, betadist = 'uniform')
            betas[sidx, ns_idx, bidx, :] = beta.ravel()
            for rep in range(reps):

                # Generate data
                start_time = time.time()
                sigma = block_covariance(correlation = correlation, block_size = b)
                X, X_test, y, y_test = gen_data(n_samples = ns, n_features = n_features, 
                    covariance = block_covariance(correlation = correlation), beta = beta)
                
                # Fit using the various estimation scores
                uoi_r2 = UoI_ElasticNet(
                normalize=True,
                n_boots_sel=48,
                n_boots_est=48,
                estimation_score='r2',
                warm_start = False)
                
                uoi_r2.fit(X, y.ravel())
                
                uoi_BIC = UoI_ElasticNet(
                normalize=True,
                n_boots_sel=48,
                n_boots_est=48,
                estimation_score='BIC',
                warm_start = False)
                
                uoi_BIC.fit(X, y.ravel())

                uoi_AIC = UoI_ElasticNet(
                normalize=True,
                n_boots_sel=48,
                n_boots_est=48,
                estimation_score='AIC',
                warm_start = False)
                
                uoi_AIC.fit(X, y.ravel())

                # Fit using the various estimation scores
                uoi_AICc = UoI_ElasticNet(
                normalize=True,
                n_boots_sel=48,
                n_boots_est=48,
                estimation_score='AICc',
                warm_start = False)
                
                try:
                    uoi_AICc.fit(X, y.ravel())
                    AICc_scores[rep, sidx, ns_idx, bidx] = r2_score(y_test, uoi_AICc.predict(X_test))
                    AICc_fn[rep, sidx, ns_idx, bidx] = np.count_nonzero(beta[uoi_AICc.coef_ == 0, 0])
                    AICc_fp[rep, sidx, ns_idx, bidx] = np.count_nonzero(uoi_AICc.coef_[beta.ravel() == 0])
                except:
                    AICc_scores[rep, sidx, ns_idx, bidx] = np.nan
                    
                # Record nominal scores and false positives/negatives
                r2_scores[rep, sidx, ns_idx, bidx] = r2_score(y_test, uoi_r2.predict(X_test))
                r2_fn[rep, sidx, ns_idx, bidx] = np.count_nonzero(beta[uoi_r2.coef_ == 0, 0])
                r2_fp[rep, sidx, ns_idx, bidx] = np.count_nonzero(uoi_r2.coef_[beta.ravel() == 0])

                BIC_scores[rep, sidx, ns_idx, bidx] = r2_score(y_test, uoi_BIC.predict(X_test))
                BIC_fn[rep, sidx, ns_idx, bidx] = np.count_nonzero(beta[uoi_BIC.coef_ == 0, 0])
                BIC_fp[rep, sidx, ns_idx, bidx] = np.count_nonzero(uoi_BIC.coef_[beta.ravel() == 0])

                AIC_scores[rep, sidx, ns_idx, bidx] = r2_score(y_test, uoi_AIC.predict(X_test))
                AIC_fn[rep, sidx, ns_idx, bidx] = np.count_nonzero(beta[uoi_AIC.coef_ == 0, 0])
                AIC_fp[rep, sidx, ns_idx, bidx] = np.count_nonzero(uoi_AIC.coef_[beta.ravel() == 0])

                print(time.time() - start_time)

results['betas'] = betas
results['r2_scores'] = r2_scores
results['r2_fp'] = r2_fp
results['r2_fn'] = r2_fn

results['BIC_scores'] = BIC_scores
results['BIC_fp'] = BIC_fp
results['BIC_fn'] = BIC_fn

results['AIC_scores'] = AIC_scores
results['AIC_fp'] = AIC_fp
results['AIC_fn'] = AIC_fn

results['AICc_scores'] = AICc_scores
results['AICc_fp'] = AICc_fp
results['AICc_fn'] = AICc_fn

results.close()
print('Total runtime: %f' %  (time.time() - total_start))
