import os, glob
import importlib
import h5py
import pickle
import numpy as np
import pandas as pd
import pdb


# Common postprocessing operations on a single data file
def postprocess(data_file, params):

	# beta and r2_true is already in correct format

	# beta_hats are originally in format reps, correlations, selection_thres_mins, n_features
	# need to go to (reps, n_features)

	# fn, fp, r2 are in shape (reps, correlations, selection_thres_mins)
	
	# Ensure that selection_thres_mins and correlations are numpy arrays

	if not isinstance(params.selection_thres_mins, np.ndarray):
		if np.isscalar(params.selection_thres_mins):
			params.selection_thres_mins = np.array([params.selection_thres_mins])
		else:
			params.selection_thres_mins = np.array(params.selection_thres_mins)

	if not isinstance(params.correlations, np.ndarray):
		if np.isscalar(params.correlations):
			params.correlations = np.array([params.correlations])
		else:
			params.correlations = np.array(params.correlations)

	data_list = []

	for cidx, corr in enumerate(params.correlations):
		# Create a dictionary to append to the main data frame
		data_dict = {}
		# Parameters to store
		try:
			data_dict['sparsity'] = params.sparsity
			data_dict['block_size'] = params.block_size
			data_dict['betadist'] = params.betadist
			data_dict['correlation'] = corr
			data_dict['beta'] = data_file['beta'][:]
			data_dict['r2_true'] = data_file['r2_true'][:, cidx]
			data_dict['r2'] = data_file['r2'][:, cidx]
			data_dict['fn'] = data_file['fn'][:, cidx]
			data_dict['fp'] = data_file['fp'][:, cidx]
			data_dict['beta_hats'] = data_file['beta_hats'][:, cidx, :]
		except:
			pdb.set_trace()

		data_list.append(data_dict)

	return data_list


# Postprocess a single file and parameter file pair
def postprocess_file(data_file, param_file):
	# Load the parameters
	params = importlib.import_module(param_file)

	# Load the data

	file = h5py.File(data_file, 'r')

	data_list = postprocess(file, params)

	# Copy to dataframe
	dataframe = pd.DataFrame(data_list)

	return dataframe


# Postprocess an entire directory of data, will assume standard nomenclature of
# associated parameter files
def postprocess_dir(dirname):
	os.chdir('data/%s' % dirname)
	# Collect all .h5 files
	data_files = glob.glob('*.h5')


	# List to store all data
	data_list = []

	for data_file in data_files:
		# Load the corresponding parameter file
		params = importlib.import_module('%s_params' % data_file.split('.h5')[0])

		# Load the data
		file = h5py.File(data_file, 'r')

		data_list.extend(postprocess(file, params))

	# Copy to dataframe
	dataframe = pd.DataFrame(data_list)

	# Save to file
	# f = open('all', 'wb')

	# pickle.dump(dataframe, f)
	return dataframe