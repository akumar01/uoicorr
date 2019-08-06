import numpy as np
from utils import FNR, FPR, selection_accuracy, estimation_error
from sklearn.metrics import r2_score, mean_squared_error
from mpi_utils.ndarray import Gatherv_rows
import pdb

# Categorize results according to selection method
def init_results_container(selection_methods, fields, num_tasks, n_features, n_reg_params = 1):

    # Use a nested dict
    results = {selection_method: {} for selection_method in selection_methods}

    for selection_method in selection_methods:
        # For all except beta_hat, initialize arrays of size num_tasks
        for field in fields[selection_method]:
            results[selection_method][field] = np.zeros(num_tasks)
        if 'beta_hats' in fields[selection_method]:
            results[selection_method]['beta_hats'] = np.zeros((num_tasks, n_features))
        if 'reg_param' in fields[selection_method]:
            results[selection_method]['reg_param'] = np.zeros((num_tasks, n_reg_params))

    return results

# Define here how to calculate various quantities of interest. Stay consistent 
# with field names
def calc_result(X, X_test, y, y_test, beta, field, exp_results): 

    y = y.ravel()
    y_test = y_test.ravel()
    beta = beta.ravel()
    beta_hat = exp_results['coefs'].ravel()

    if field == 'beta_hats':

        result = beta_hat

    elif field == 'FNR':

        result = FNR(beta.ravel(), beta_hat.ravel())

    elif field == 'FPR':

        result = FPR(beta.ravel(), beta_hat.ravel())

    elif field == 'sa':

        result = selection_accuracy(beta.ravel(), beta_hat.ravel())

    elif field == 'ee':

        result, _ = estimation_error(beta.ravel(), beta_hat.ravel())

    elif field == 'median_ee':

        _, result = estimation_error(beta.ravel(), beta_hat.ravel())

    elif field == 'r2':

        result = r2_score(y_test, X_test @ beta)

    elif field == 'MSE':

        result = mean_squared_error(y_test, X_test @ beta)

    elif field == 'reg_param': 

        result = exp_results['reg_param']

    elif field == 'oracle_penalty':

        result = exp_results['oracle_penalty']

    else:
        raise ValueError('field type not understood')

    return result

# Calculate the best result along a solution path
def calc_path_result(X, X_test, y, y_test, beta, field, exp_results): 

    y = y.ravel()
    y_test = y_test.ravel()
    beta = beta.ravel()
    beta_hat = exp_results['coefs']

    if field == 'FNR':

        result = np.nanmin(FNR(beta, beta_hat))

    elif field == 'FPR':

        result = np.nanmin(FPR(beta, beta_hat))

    elif field == 'sa':

        result = np.nanmax(selection_accuracy(beta, beta_hat))

    elif field == 'ee':

        result, _ = estimation_error(beta, beta_hat)
        result = np.nanmin(result)

    elif field == 'median_ee':

        _, result = estimation_error(beta, beta_hat)
        result = np.nanmin(result)

    # elif field == 'r2':

    #     result = r2_score(y_test, X_test @ beta)


    # elif field == 'MSE':

    #     result = mean_squared_error(y_test, X_test @ beta)

    return result

# Gather each entry of results and return the final dictionary
def gather_results(results, comm): 

    gathered_results = {}

    for selection_method in results.keys():
        gathered_results[selection_method] = {}
        for field in results[selection_method].keys():

            value = Gatherv_rows(results[selection_method][field], comm, root = 0)
    
            gathered_results[selection_method][field] = value 

    return gathered_results
