import sys, os
import argparse
import glob
import pdb

# Resubmit all the jobs in a given directory or a file
parser = argparse.ArgumentParser()
parser.add_argument('directory', default = None)
a = parser.parse_args()

if a.directory: 
	# Collect all .h5 files
	sbatch_files = glob.glob('uoicorr/%s/sbatch*' % a.directory)
	for sbf in sbatch_files:
		os.system('sbatch %s' % sbf)		

