import sys, os
from datetime import datetime
import subprocess
import shlex
import pdb
import itertools
import glob
import argparse
import json
import importlib
import subprocess
import numpy as np

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
		arg['results_file'] = results_files[i]
		arg_file = '%s/%s_params.json' % (jobdir, jobnames[i])
		with open(arg_file, 'w') as f:
			json.dump(arg, f)
			f.close()
		# Strip the .py from the end
		arg_files.append(arg_file)
	return arg_files 

if __name__ == '__main__':
	# Create and execute an sbatch script for each desired job 
	# For each job, need to specify the python script to be used 
	# and the relevant arguments. These values are specified as a list
	# of dictionaries

	# Command line arguments
	parser = argparse.ArgumentParser()
	
	# Param file from which to create job scripts
	parser.add_argument('param_file', default=None)

	# Create job files without actually submitting them
	parser.add_argument('-t', '--test', action='store_true')
	# Submit only the first job
	parser.add_argument('-f', '--first_only', action='store_true')
	# Edit the 'reps' parameter to equal 1
	parser.add_argument('-s', '--single_rep', action='store_true')

	# Run the job specified by iidx interactively instead of submitting to sbatch
	parser.add_argument('-i', '--interactive', action='store_true')
	parser.add_argument('--iidx', default = 0)

	cmd_args = parser.parse_args()


	# Load param file
	
	path, name = cmd_args.param_file.split('/')
	sys.path.append(path)
	params = importlib.import_module(name)
	f = open('%s.py' % cmd_args.param_file, 'r')
	fcontents = f.read()

	script_dir = params.script_dir

	root_dir = params.root_dir

	jobdir = params.jobdir

	jobdir = '%s/%s' % (root_dir, jobdir)

	if not os.path.exists(jobdir):
		os.makedirs(jobdir)

	script = params.script
	iter_params = params.iter_params
	iter_keys = list(iter_params.keys())
	comm_params = params.comm_params
	desc = params.desc

	# Change the reps to 1 if flagged
	if cmd_args.single_rep:
		if 'reps' in iter_params.keys():
			iter_params['reps'] = [1]
		elif 'reps' in comm_params.keys():
			comm_params['reps'] = 1
		else:
			print('Warning! No reps parameter found')

	jobnames =  []
	args = []

	# Iterate over all combinations of parameters in iter_params and combine them
	# with comm_params to produce a unique argument dictionary for each individual job
	for i, arg_comb in enumerate(itertools.product(*list(iter_params.values()))):
		arg = {}
		for j in range(len(arg_comb)):
			arg[iter_keys[j]] = arg_comb[j]			
		for key, value in comm_params.items():
			arg[key] = value

		args.append(arg)
		jobnames.append('job%d' % i)

	hostname = subprocess.check_output(['hostname'])

	cont = input("You are about to submit %d jobs, do you want to continue? [0/1]" % len(jobnames))
	if cont:

		results_files = ['%s/%s.h5' % (jobdir, jobname) for jobname in jobnames]
		# Write the arguments to file
		arg_files = write_args_to_file(args, results_files, jobnames, jobdir)
		# Generate an interable containing the script name and the parameter file name
		jobs = [{'script': script, 'arg_file' : arg_file} for arg_file in arg_files]

		# Ensure we aren't accidentally duplicating/overwriting existing jobs
		# validate_jobs(jobdir, jobnames)

		if cmd_args.interactive:
			# Run the job interactively to verify things are working ok
			if cmd_args.iidx == 'all':
				iidx = np.arange(len(jobs), dtype = np.int_)
			else:
				iidx = [int(cmd_args.iidx)]
			for idx in iidx:
				os.system('python %s/%s %s' % (script_dir, jobs[idx]['script'], jobs[idx]['arg_file']))
			sys.exit()

		# Log all job details
		log_file = open('%s/log.txt' % jobdir, 'w')
		log_file.write('Jobs submitted at ' + "{:%H:%M:%S, %B %d, %Y}".format(datetime.now()) + '\n\n\n')
		log_file.write(fcontents)

		# Set this environment variable to prevent crashses
		os.system('export HDF5_USE_FILE_LOCKING=FALSE')

		# Write an sbatch script for each job
		for i, job in enumerate(jobs):

			sbname = '%s/sbatch%d.sh' % (jobdir, i)
			with open(sbname, 'w') as sb:
				# Arguments common across jobs
				sb.write('#!/bin/bash\n')
				sb.write('#SBATCH -q shared\n')
				sb.write('#SBATCH -n 1\n')
				sb.write('#SBATCH -t %s\n' % params.job_time)

				sb.write('#SBATCH --job-name=%s\n' % jobnames[i])
				sb.write('#SBATCH --out=%s/%s.o\n' % (jobdir, jobnames[i]))
				sb.write('#SBATCH --error=%s/%s.e\n' % (jobdir, jobnames[i]))
				sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
				sb.write('#SBATCH --mail-type=FAIL\n')
				# Load python and any other necessary modules
				# sb.write('module load python/3.6-anaconda-4.4\n')
				
				# Failsafe in case we have alerady loaded python. Loading two
				# different python modules will cause an error

				# Work with out own Anaconda environment
				# To make this work, we had to add some paths to our .bash_profile.ext
				sb.write('source activate nse\n')

				# script(s) to actually run
				if 'cori'.encode() in hostname:
					sb.write('srun -C haswell python3 -u  %s/%s %s' 
						% (script_dir, job['script'], job['arg_file']))
				else:
					sb.write('srun python3 -u %s/%s %s' 
						% (script_dir, job['script'], job['arg_file']))
				sb.close()
				
			# Change permissions
			os.system('chmod u+x %s' % sbname)
			if not cmd_args.test:
				if not cmd_args.first_only or i == 0:
					# Submit the job
					os.system('sbatch %s ' % sbname)

		log_file.close()
