n_features = 50
n_samples = 150

sigma = gen_covariance(n_features, 0, n_features, 1, 0)
beta = gen_beta2(n_features, n_features, sparsity = 0.2, betawidth = np.inf)
X, X_test, y, y_test, ss = gen_data(n_samples, n_features, kappa = 100, 
									covariance = sigma, beta = beta)