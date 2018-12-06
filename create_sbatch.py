import os
from datetime import datetime
import subprocess
import shlex
import pdb

# Make sure we aren't mistakenly submitting jobs with incorrect parameters
def validate_jobs(jobs):
	# Ensure that result_files are all unique
	rfs = []
	for job in jobs:
		args = shlex.split(job['args'])
		rfile = [a for a in args if 'results_file' in a][0].split('results_file=')[1]
		rfs.append(rfile)
	assert len(frozenset(rfs)) == len(rfs), 'Not all results files are unique!'

if __name__ == '__main__':
	# Create and execute an sbatch script for each desired job 
	# For each job, need to specify the python script to be used 
	# and the relevant arguments. These values are specified as a list
	# of dictionaries

	script_dir = '/global/homes/a/akumar25'

	root_dir = '/global/homes/a/akumar25/uoicorr'

	jobdir = '12052018'

	d = '%s/%s' % (root_dir, jobdir)

	if not os.path.exists(d):
		os.makedirs(d)

	jobnames = ['sparse1', 'sparse08', 'sparse06', 'sparse04', 'sparse02']
	jobs = [{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5 --kappa=0.1 --sparsity=1' % (d, jobnames[0])},
			{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5 --kappa=0.1 --sparsity=08' % (d, jobnames[1])},
			{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5 --kappa=0.1 --sparsity=06' % (d, jobnames[2])},
			{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5 --kappa=0.1 --sparsity=04' % (d, jobnames[3])},
			{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5 --kappa=0.1 --sparsity=02' % (d, jobnames[4])}]
	validate_jobs(jobs)
	# Log stuff
	log_file = open('%s/log.txt' % d, 'w')
	log_file.write('Jobs submitted at ' + "{:%H:%M:%S, %B %d, %Y}".format(datetime.now()) + '\n\n\n')

	# Write an sbatch script for each job
	for i, job in enumerate(jobs):
		log_file.write('Job name: %s\n' % jobnames[i])
		for key, val in job.items():
			log_file.write('%s: %s\n'  %(key, val))
		log_file.write('\n\n')

		sbname = '%s/sbatch%d.sh' % (d, i)
		with open(sbname, 'w') as sb:
			# Arguments common across jobs
			sb.write('#!/bin/bash\n')
			sb.write('#SBATCH -q shared\n')
			sb.write('#SBATCH -n 1\n')
			sb.write('#SBATCH -t 20:00:00\n')

			sb.write('#SBATCH --job-name=%s\n' % jobnames[i])
			sb.write('#SBATCH --out=%s/%s.o\n' % (d, jobnames[i]))
			sb.write('#SBATCH --error=%s/%s.e\n' % (d, jobnames[i]))
			sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
			sb.write('#SBATCH --mail-type=FAIL\n')
			# Load python and any other necessary modules
			sb.write('module load python/3.6-anaconda-4.4\n')
			# script(s) to actually run
			sb.write('srun -C haswell python3  %s/%s %s' % (script_dir, job['script'], job['args']))
			sb.close()

		# Change permissions
		os.system('chmod u+x %s' % sbname)

		# Submit the job
		os.system('sbatch %s ' % sbname)

	log_file.close()