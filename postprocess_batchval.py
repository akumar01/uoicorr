import h5py
import numpy as np
import pandas as pd
from job_manager import grab_files


# Merge together results from within a single fold
def postprocess_fold(files, fold):

	# Get the fields that need to be concatenated
	with h5py.File(files[0], 'r') as f:
		fields = list(f.keys())

	data_dict = {}
	for field in fields:
		data_dict[field] = []

	for file in files:
		with h5py.File(file, 'r') as f:
			for field in fields:
				data_dict[field].append(f[field][:])

	data_dict['fold'] = fold

	return data_dict

def postprocess_dir(jobdir, dataset, exp_types = None):

	# Grab the set of .dat files associated with the given dataset and exp_type (if given)
	if exp_types is None:
		exp_types = ['CV_Lasso', 'EN', 'UoILasso', 'UoIElasticNet']

	data_list = []

	for exp_type in exp_types: 
		datfiles = grab_files('%s/%s' % (jobdir, dataset), '*.dat', exp_type)

		# Organize with respect to fold and column set:
		datfiles, folds, cols = group_datfiles(datfiles)

		# Results within folds:

		for i in range(len(datfiles)):
			d = postprocess_fold(datfiles[i], folds[i])
			data_list.extend(d)			

	dataframe = pd.DataFrame(data_list)
	print(dataframe.shape)
	return dataframe

# Given a list of paths to data files, return a nested list of dimension (n_folds, n_column_sets)
def group_datfiles(datfiles):

	# Get the fold of every data file
	folds = [int(d.split('_fold')[1].split('_col')[0]) for d in datfiles]
	# Get the column set index of every data file
	cols = [int(d.split('.dat')[0].split('col')[1]) for d in datfiles]

	unique_folds = np.unique(folds)
	unique_cols = np.unique(cols)

	grouped_files = []

	for i, fold in enumerate(unique_folds):
		grouped_files.append([])
		for j, col in enumerate(unique_cols):
			grouped_files[i].append([])

			# Grab all the data files that match this combination of fold and column set
			matching_files = [datfiles[k] for k in range(len(datfiles)) if folds[k] == fold
							  and cols[k] == col]
			grouped_files[i][j].extend(matching_files)

	return grouped_files, folds, cols