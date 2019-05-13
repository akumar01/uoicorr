import networkx as nx
from networkx.algorithms.cluster import clustering
from job_manager import grab_files, run_
import os
import sys
import importlib
from glob import glob
# Calculate clustering coefficients and assortativity for all covariance matrices. Submit
# each calculation as a separate batch job


def create_sbatch_files(jobdir, cov_params):

    for i, cov_param in enumerate(cov_params):
        sbname = '%s/sbatch%d.sh' % (jobdir, i)
        jobname = 'graph_calc_%d' % i

        script_dir = '/global/homes/a/akumar25/repos/uoicorr'
        script = 'graph_calc.py'
        results_file = '%s/job%d.dat' % (jobdir, i)

        with open(sbname, 'w') as sb:
            # Arguments common across jobs
            sb.write('#!/bin/bash\n')
            sb.write('#SBATCH -q shared\n')
            sb.write('#SBATCH -n 1\n')
            sb.write('#SBATCH -t 03:00:00\n')
            sb.write('#SBATCH --job-name=%s\n' % jobname)
            sb.write('#SBATCH --out=%s/%s.o\n' % (jobdir, jobname))
            sb.write('#SBATCH --error=%s/%s.e\n' % (jobdir, jobname))
            sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
            sb.write('#SBATCH --mail-type=FAIL\n')

            # Work with out own Anaconda environment
            sb.write('source ~/anaconda3/bin/activate\n')
            sb.write('source activate nse\n')
           
            sb.write('srun python3 -u %s/%s %s -c %f -b %d -L %d -t %f' 
                            % (script_dir, script, results_file, cov_param['correlation'],
                            cov_param['block_size'], cov_param['L'], cov_param['t']))

def create_job_structure(submit_file, jobdir):

    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    path, name = submit_file.split('/')
    sys.path.append(path)
    args = importlib.import_module(name)

    cov_params = args.cov_params

    create_sbatch_files(jobdir, cov_params)

def run_jobs(jobdir, constraint, run = False):

    run_files = glob('%s/*.sh' % jobdir)
    print('Submitting %d jobs' % len(run_files))

    for run_file in run_files:
        # Need to open up the file and include the particular constraint
        # On Edison - remove any constraints
        # On Cori - can have either KNL or Haswell
        f = open(run_file, 'r')
        contents = f.readlines()
        f.close()

        constraint_string = [s for s in contents if '--constraint' in s]

        # First remove any existing constraint strings
        if len(constraint_string) > 0:
            for c in constraint_string:
                contents.remove(c)
        
        # Add constraint to the 2nd line (index 1) so it is the first
        # thing after #/bin/bash
        if constraint == 'haswell':
            contents.insert(1, '#SBATCH --constraint=haswell\n')
          
        if constraint == 'knl':
            contents.insert(1, '#SBATCH --constraint=knl\n')                  
        
        f = open(run_file, "w")
        contents = "".join(contents)
        f.write(contents)
        f.close()
        
        if run:
            run_(run_file)
