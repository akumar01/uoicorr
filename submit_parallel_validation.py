import argparse
import h5py
from sklearn.model_selection import KFold
import itertools
import os

p = argparse.ArgumentParser()
parser.add_argument('datapath')
parser.add_argument('jobdir')
parser.add_argument('dataset')

args = parser.parse_args()


# Create directory structure:
if not os.path.exists(args.jobdir):
	os.makedirs(jobdir)

# TO DO: Decide which form of the data to load in this file
# Then, for each dataset, generate the folds in the proper way
# and then save away the data, and generate all the necessary 
# sbatch files. We have opted for shared queue as opposed to
# trying to use MPI

if args.dataset == 'A1':
	from neuropacks.neuropacks import ECOG
	
	if not os.path.exists('%s/%s' % (args.jobdir, args.dataset)):
		os.makedirs('%s/%s' % (args.jobdir, args.dataset))

	ecog = ECOG(args.datapath)
	response_matrix = ecog.get_response_matrix(bounds = (40, 60),
												band ='HG')

	# Normalize
	response_matrix = response_matrix - np.mean(response_matrix, axis = 0)
	response_matrix = response_matrix/np.std(response_matrix, axis = 0)

	# Partition into folds
	kfold = KFold(n_splits = 5, shuffle = True, random_state = 25)
	train_data_folds = []
	test_data_folds = []
	for train_idxs, test_idxs in kfold.split(response_matrix):
		train_data_folds.append(response_matrix[train_idxs, :])
		test_data_folds.append(response_matrix[test_idxs, :])

	
