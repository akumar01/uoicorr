import sys, os
import argparse
import glob
import pdb
import json


# Resubmit all the jobs in a given directory or a file
def resubmit_jobs(directory, jobnums = None):
	# Collect all sbatch files
	sbatch_files = glob.glob('uoicorr/%s/sbatch*' % directory)
	for sbf in sbatch_files:
		if jobnums is not None:
			jobnum = int(sbf.split('.sh')[0].split('sbatch')[1])
			if jobnum in jobnums:
				os.system('sbatch %s' % sbf)		
		else:
				os.system('sbatch %s' % sbf)		
			
# Change a parameter given by key to value in jobs given by jobnums. If jobnums is none, the change
# will be applied to all jobs in directory. Flag sbatch tells whether we should look for parameter
# key in the job_param file or the sbatch file (for example to change the job clock time, we would
# set sbatch = True)
def change_param(key, value, directory, sbatch = False, jobnums = None):

	# Grab all files
	if sbatch:
		files = glob.glob('uoicorr/%s/sbatch*' % directory)

		for file in files:
			if jobnums is not None:
				jobnum = int(file.split('.sh')[0].split('sbatch')[1])

				if jobnum in jobnums:
					edit_sbatch(file, key, value)	

			else:
				edit_sbatch(file, key, value)

	else:
		files = glob.glob('uoicorr/%s/*.json' % directory)


		for file in files:
			if jobnums is not None:
				jobnum = int(file.split('.sh')[0].split('sbatch')[1])

				if jobnum in jobnums:
					with open(file, 'r+') as f:
					# change parameter and add sbatch to list
						params = json.load(f)
					params[key] = value
					with open(file, 'w') as f:
						json.dump(params, f)
			else:
				with open(file, 'r+') as f:
				# change parameter and add sbatch to list
					params = json.load(f)
				params[key] = value
				with open(file, 'w') as f:
					json.dump(params, f)

# Given an sbatch file, find parameter given by flag and set to value
def edit_sbatch(file, flag, value):
	with open(file, 'r') as fobject:
		fcontent = fobject.read()
	paramstring = '#SBATCH %s' % flag
	start_loc = fcontent.find(paramstring)
	if start_loc != -1:
		end_loc = start_loc + len(paramstring) + 1

		endofline = fcontent.find('\n', end_loc)

		current_value = fcontent[end_loc:endofline]
		fcontent = fcontent.replace(current_value, value)
		with open(file, 'w') as fobject:
			fobject.write(fcontent)
	else:
		print('Could not find flag in sbatch file')

