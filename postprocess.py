import os, glob, sys
import json
import h5py
import pickle
import numpy as np
import pandas as pd
import itertools
import importlib
import pdb


# Common postprocessing operations on a single data file
def postprocess(data_file, params, arg_flags = None):

    # beta and r2_true is already in correct format

    # beta_hats are originally in format reps, correlations, n_features
    # need to go to (reps, n_features)

    # fn, fp, r2 are in shape (reps, correlations)

    data_list = []

    # Wrap dictionary in a list
    if params['cov_type'] == 'interpolate':
        if type(params['cov_params']) != list:
            cov_params = [params['cov_params']]
        else:
            cov_params = params['cov_params']
    else:
        cov_params = params['cov_params']

    if arg_flags is not None:
        arg_found = False
        for key, value in arg_flags.items():
            if key in list(params.keys()):
                if params[key] == value:
                    arg_found = True
    else:
        arg_found = True

    if not arg_found:
        return

    for covidx, cov_param in enumerate(cov_params):

        data_dict = {} 

        data_dict = params.copy()
        data_dict['cov_params'] = cov_param
        # For standard uoicorr_base experiments
        try:
            data_dict['betas'] = data_file['betas'][:]
        except:
            try:
                data_dict['betas'] = data_file['beta'][:]
            except:
                pass
            
        data_dict['r2_true'] = data_file['r2_true'][:, covidx]
        data_dict['r2'] = data_file['r2'][:, covidx]
        data_dict['fn'] = data_file['fn'][:, covidx]
        data_dict['fp'] = data_file['fp'][:, covidx]
        data_dict['beta_hats'] = data_file['beta_hats'][:, covidx, :]

        if 'BIC' in list(data_file.keys()):
            data_dict['BIC'] = data_file['BIC'][:, covidx]
        if 'AIC' in list(data_file.keys()):
            data_dict['AIC'] = data_file['AIC'][:, covidx]
        if 'AICc' in list(data_file.keys()):
            data_dict['AICc'] = data_file['AICc'][:, covidx]
        if 'FNR' in list(data_file.keys()):
            data_dict['FNR'] = data_file['FNR'][:, covidx]
            data_dict['FPR'] = data_file['FPR'][:, covidx]
            data_dict['sa'] = data_file['sa'][:, covidx]
            data_dict['ee'] = data_file['ee'][:, covidx]
            data_dict['median_ee'] = data_file['median_ee'][:, covidx]
                        
        # For est comparison experiments
        # data_dict['betas'] = data_file['betas'][:]
        # data_dict['r2_scores'] = data_file['r2_scores'][:]
        # data_dict['r2_fp'] = data_file['r2_fp'][:]
        # data_dict['r2_fn'] = data_file['r2_fn'][:]

        # data_dict['BIC_scores'] = data_file['BIC_scores'][:]
        # data_dict['BIC_fp'] = data_file['BIC_fp'][:]
        # data_dict['BIC_fn'] = data_file['BIC_fn'][:]

        # data_dict['AIC_scores'] = data_file['AIC_scores'][:]
        # data_dict['AIC_fp'] = data_file['AIC_fp'][:]
        # data_dict['AIC_fn'] = data_file['AIC_fn'][:]

        # data_dict['AICc_scores'] = data_file['AICc_scores'][:]
        # data_dict['AICc_fp'] = data_file['AICc_fp'][:]
        # data_dict['AICc_fn'] = data_file['AICc_fn'][:]

        data_list.append(data_dict)
    return data_list



# Postprocess a single file and parameter file pair
def postprocess_file(data_file, param_file):
    # Load the parameters
    with open(param_file) as f:
        # Load the corresponding parameter file
        params = json.load(f)

    # Load the data

    file = h5py.File(data_file, 'r')

    data_list = postprocess(file, params)

    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)

    return dataframe


# Postprocess an entire directory of data, will assume standard nomenclature of
# associated parameter files
# old_format: back when we were using the .py param files
# skip_bad: Skip over files that cannot be processed without raising errors
# arg_flag: Only return dataframes for data files that match arg_flag {key: value}
def postprocess_dir(dirname, old_format = False, skip_bad = False, arg_flags = None):
    # Collect all .h5 files
    data_files = glob.glob('%s/*.h5' % dirname)
    # List to store all data
    data_list = []

    if old_format:
        sys.path.append(dirname)

    for data_file in data_files:
        if old_format:
            module_name = os.path.split(data_file)[1].split('.h5')[0]
            params = importlib.import_module('%s_params' % module_name)
            params = dict_from_module(params)
            params['cov_params'] = params.pop('correlations')
            # Add necessary additional fields to parameter
            params['cov_type'] = 'block'
        else:
            with open('%s_params.json' % data_file.split('.h5')[0], 'r') as f:
                # Load the corresponding parameter file
                params = json.load(f)
        try:
            # Load the data
            file = h5py.File(data_file, 'r')
        except:
            continue
        
        # Choose whether to ignore errors:      
        if skip_bad:
            try:
                d = postprocess(file, params, arg_flags)
                if d is not None:
                    data_list.extend(d)
            except:
                continue
        else:
            d = postprocess(file, params, arg_flags)
            if d is not None:
                data_list.extend(d)

    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)

    # Save to file
    # f = open('all', 'wb')

    # pickle.dump(dataframe, f)
    print(dataframe.shape)
    return dataframe

# util function to deal with legacy usage of .py param files
def dict_from_module(module):
    context = {}
    for setting in dir(module):
        if not '__' in setting:
            context[setting] = getattr(module, setting)

    return context
