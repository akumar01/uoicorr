import os
from datetime import datetime
import subprocess

# Create and execute an sbatch script for each desired job 
# For each job, need to specify the python script to be used 
# and the relevant arguments. These values are specified as a list
# of dictionaries

script_dir = '/g\lobal/homes/a/akumar25'

root_dir = '/global/homes/a/akumar25/uoicorr'

jobdir = '11282018'

d = '%s/%s' % (root_dir, jobdir)

if not os.path.exists(d):
	os.makedirs(d)

jobnames = ['block_BIC', 'block_AIC']
jobs = [{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5' % (d, jobnames[0])},
		{'script': 'uoicorr_block.py', 'args': '--results_file=%s/%s.h5 --est_score=AIC' % (d, jobnames[1])}]

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
		sb.write('#SBATCH -t 05:00:00\n')

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