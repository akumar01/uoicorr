import numpy as np
from utils import FNR, FPR, selection_accuracy, estimation_error
from sklearn.metrics import r2_score, mean_squared_error
from mpi_utils.ndarray import Gatherv_rows
import pdb

# Categorize results according to selection method
def init_results_container(selection_methods):

    # Use a nested dict
    results = {selection_method: {} for selection_method in selection_methods}

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

        result = r2_score(y_test, X_test @ beta_hat)

    elif field == 'MSE':

        result = mean_squared_error(y_test, X_test @ beta_hat)

    elif field == 'reg_param': 
        # Some algorithms (i.e. UoI) have no 'selected' reg param
        if 'reg_param' in list(exp_results.keys()):
            result = exp_results['reg_param']
        else:
            result = np.nan

    elif field == 'oracle_penalty':
        result = exp_results['oracle_penalty']

    elif field == 'effective_penalty':
        result = exp_results['effective_penalty']

    # Will need to investigate how this is handled (ndarray)
    elif field == 'sparsity_estimates':
        result = exp_results['sparsity_estimates']

    # record oracle performance (from aBIC)
    elif field == 'oracle_sa':
        result = selection_accuracy(beta.ravel(), 
                                     exp_results['oracle_coefs'].ravel())

    elif field == 'oracle_FNR': 

        result = FNR(beta.ravel(), 
                      exp_results['oracle_coefs'].ravel())

    elif field == 'oracle_FPR':

        result = FPR(beta.ravel(), 
                      exp_results['oracle_coefs'].ravel())

    elif field == 'oracle_ee':
        result, _ = estimation_error(beta.ravel(), exp_results['oracle_coefs'].ravel())

    elif field == 'oracle_r2':

        result = r2_score(y_test, X_test @ exp_results['oracle_coefs'])

    elif field == 'oracle_MSE':

        result = mean_squared_error(y_test, X_test @ exp_results['oracle_coefs'])

    else:
        raise ValueError('field type %s not understood' % field)

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
