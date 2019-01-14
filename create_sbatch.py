import sys, os
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse


# Make sure we aren't mistakenly submitting jobs with incorrect parameters
def validate_jobs(jobdir, jobnames):
	# Ensure that no existing jobs will be overwritten
	for jobname in jobnames:
		assert not glob.glob('%s*' % jobname), 'Not all jobs are unique!'		


# Write longer arguments to file. jobdir should be the directory
# that the job files go into (global path) 
# args is a dictionary containing the arguments
def write_args_to_file(args, results_files, jobnames, jobdir):
	arg_files = []
	for i, arg in enumerate(args):
		arg_file = '%s/%s_params.py' % (jobdir, jobnames[i])
		with open(arg_file, 'w') as f:
			for key, value in arg.items():
				f.write('{0} = {1}\n'.format(key, value))
			f.write('results_file = %s' % results_files[i])
			f.close()
		# Strip the .py from the end
		arg_files.append(arg_file.split('.py')[0])
	return arg_files 

if __name__ == '__main__':
	# Create and execute an sbatch script for each desired job 
	# For each job, need to specify the python script to be used 
	# and the relevant arguments. These values are specified as a list
	# of dictionaries

	# Command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-test', action='store_true')
	parser.add_argument('-first_only', action='store_true')
	
	cmd_args = parser.parse_args()
	

	script_dir = '/global/homes/a/akumar25'

	root_dir = '/global/homes/a/akumar25/uoicorr'

	jobdir = '01132019c'

	jobdir = '%s/%s' % (root_dir, jobdir)

	if not os.path.exists(jobdir):
		os.makedirs(jobdir)

	# Specify script to use:
	script = 'elasticnet_block.py'

	# List the set of arguments to the script(s) that will be iterated over
	iter_params = {'sparsity' : [1., 0.8, 0.6, 0.4, 0.2], 'block_size': [6, 12, 20, 30]}
	iter_keys = list(iter_params.keys())

	# List arguments that will be held constant across all jobs:
	comm_params = {'kappa' : 0.3, 'n_features' : 60,
	'correlations' : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
	'est_score': '\'r2\'', 'reps' : 50, 'selection_thres_mins' : [1.0],
	'n_samples' : 60 * 5, 'betadist':'\'uniform\''}

	# Parameters for ElasticNet
	if script == 'elasticnet_block.py' or script == 'uoien_block.py':
		comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
		comm_params['n_alphas'] = 48

	# Description of what the job is
	desc = "Re-doing 01112019b with reps set to 50 (and not 1)"
	jobnames =  []
	args = []

	for i, arg_comb in enumerate(itertools.product(*list(iter_params.values()))):
		arg = {}
		for j in range(len(arg_comb)):
			arg[iter_keys[j]] = arg_comb[j]			
		for key, value in comm_params.items():
			arg[key] = value

		args.append(arg)
		jobnames.append('job%d' % i)

	cont = input("You are about to submit %d jobs, do you want to continue? [0/1]" % len(jobnames))
	if cont:

		results_files = ['\'%s/%s.h5\'' % (jobdir, jobname) for jobname in jobnames]
		# Write the arguments to file
		arg_files = write_args_to_file(args, results_files, jobnames, jobdir)
		# Generate an interable containing the script name and the parameter file name
		jobs = [{'script': script, 'arg_file' : arg_file} for arg_file in arg_files]

		# Ensure we aren't accidentally duplicating/overwriting existing jobs
		validate_jobs(jobdir, jobnames)

		# Log all job details
		log_file = open('%s/log.txt' % jobdir, 'w')
		log_file.write('Jobs submitted at ' + "{:%H:%M:%S, %B %d, %Y}".format(datetime.now()) + '\n\n\n')
		log_file.write('Run Description: %s\n\n\n' % desc)

		# Write an sbatch script for each job
		for i, job in enumerate(jobs):
			log_file.write('Job name: %s\n' % jobnames[i])
			for key, val in job.items():
				log_file.write('%s: %s\n'  %(key, val))
			log_file.write('\n\n')

			sbname = '%s/sbatch%d.sh' % (jobdir, i)
			with open(sbname, 'w') as sb:
				# Arguments common across jobs
				sb.write('#!/bin/bash\n')
				sb.write('#SBATCH -q shared\n')
				sb.write('#SBATCH -n 1\n')
				sb.write('#SBATCH -t 10:00:00\n')

				sb.write('#SBATCH --job-name=%s\n' % jobnames[i])
				sb.write('#SBATCH --out=%s/%s.o\n' % (jobdir, jobnames[i]))
				sb.write('#SBATCH --error=%s/%s.e\n' % (jobdir, jobnames[i]))
				sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
				sb.write('#SBATCH --mail-type=FAIL\n')
				# Load python and any other necessary modules
				sb.write('module load python/3.6-anaconda-4.4\n')
				# script(s) to actually run
				sb.write('srun -C haswell python3  %s/%s --arg_file=%s' 
					% (script_dir, job['script'], job['arg_file']))
				sb.close()
				
			# Change permissions
			os.system('chmod u+x %s' % sbname)
			if not cmd_args.test:
				if not cmd_args.first_only or i == 0:
					# Submit the job
					os.system('sbatch %s ' % sbname)

		log_file.close()
