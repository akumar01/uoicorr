import os, glob
import json
import h5py
import pickle
import numpy as np
import pandas as pd
import itertools
import pdb


# Common postprocessing operations on a single data file
def postprocess(data_file, params):

	# beta and r2_true is already in correct format

	# beta_hats are originally in format reps, correlations, n_features
	# need to go to (reps, n_features)

	# fn, fp, r2 are in shape (reps, correlations)

	data_list = []
	
	for covidx, cov_param in enumerate(params['cov_params']):
		data_dict = {} 

		# beta has different length
		data_dict['beta'] = data_file['beta'][:]

		data_dict = params.copy()
		data_dict['cov_params'] = cov_param

		# For standard uoicorr_base experiments
		data_dict['betas'] = data_file['betas'][:]
		data_dict['r2_true'] = data_file['r2_true'][:, covidx]
		data_dict['r2'] = data_file['r2'][:, covidx]
		data_dict['fn'] = data_file['fn'][:, covidx]
		data_dict['fp'] = data_file['fp'][:, covidx]
		data_dict['beta_hats'] = data_file['beta_hats'][:, covidx, :]

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
def postprocess_dir(dirname):
	# Collect all .h5 files
	data_files = glob.glob('data/%s/*.h5' % dirname)

	# List to store all data
	data_list = []

	for data_file in data_files:
		with open('%s_params.json' % data_file.split('.h5')[0], 'r') as f:
			# Load the corresponding parameter file
			params = json.load(f)
		try:
			# Load the data
			file = h5py.File(data_file, 'r')
		except:
			continue
		data_list.extend(postprocess(file, params))

	# Copy to dataframe
	dataframe = pd.DataFrame(data_list)

	# Save to file
	# f = open('all', 'wb')

	# pickle.dump(dataframe, f)
	return dataframe