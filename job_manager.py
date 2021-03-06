import numpy as np
import importlib
import itertools
import pdb
import sys, os
import pickle
import time
import traceback
import pandas as pd
from glob import glob
from subprocess import check_output
from utils import gen_covariance, gen_beta2, gen_data

def chunk_list(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

# Convert all numpy datatypes to native python
# datatypes
def fix_datatypes(obj):

    # If list, recursively search through it:
    if type(obj) == list:
        for idx, sub_obj in enumerate(obj):
            obj[idx] = fix_datatypes(sub_obj)

    # If ndarray, convert to list, and then
    # recursively search through it:
    if type(obj) == np.ndarray:
        obj = obj.tolist()
        for idx, sub_obj in enumerate(obj):
            obj[idx] = fix_datatypes(sub_obj)

    # If dictionary, iterate through its values:
    if type(obj) == dict:
        for key, value in obj.items():
            # Skip this guy because it could be quite large
            # and tedious to recursively go through
            if key == 'sigma':
                continue
            else:
                obj[key] = fix_datatypes(value)

    if type(obj) == np.int_:
        obj = obj.item()
    if type(obj) == np.float_:
        obj = obj.item()
    return obj

def generate_arg_files(argfile_array, jobdir):

    paths = []
    ntasks = []

    for i, arg_ in enumerate(argfile_array):
        start = time.time()

        arg_file = '%s/master/params%d.dat' % (jobdir, i)
        paths.append(arg_file)

        # Generate the full set of data/metadata required to run the job

        sub_iter_params = arg_['sub_iter_params']
        for key in sub_iter_params:
            if not hasattr(arg_[key], '__len__'):
                arg_[key] = [arg_[key]]
            arg_[key] = list(arg_[key])

        # Complement of the sub_iter_params:
        const_keys = list(set(arg_.keys()) - set(sub_iter_params))

        const_args = {k: arg_[k] for k in const_keys}

        # Combine reps and parameters to be cycled through into a single iterable. Store
        # the indices pointing to the corresponding rep/parameter list for easy unpacking
        # later on
    
        arg_comb = arg_['reps']\
                            * list(itertools.product(*[arg_[key] for key in arg_['sub_iter_params']]))
        iter_param_list = []
        for i in range(len(arg_comb)):
            arg_dict = const_args.copy()
            [arg_dict.update({arg_['sub_iter_params'][j]: arg_comb[i][j] for j in range(len(arg_['sub_iter_params']))})]
            iter_param_list.append(arg_dict)

        ntasks.append(len(iter_param_list))

        for i, param_comb in enumerate(iter_param_list):

            if 'n_samples' in list(param_comb.keys()):
                n_samples = param_comb['n_samples']
            elif 'np_ratio' in list(param_comb.keys()):
                n_samples = int(param_comb['np_ratio'] * param_comb['n_features'])
                param_comb['n_samples'] = n_samples
            sigma = gen_covariance(param_comb['n_features'],
                                   param_comb['cov_params']['correlation'], 
                                   param_comb['cov_params']['block_size'],
                                   param_comb['cov_params']['L'],
                                   param_comb['cov_params']['t'])
            betas = gen_beta2(param_comb['n_features'], param_comb['cov_params']['block_size'],
                              param_comb['sparsity'], param_comb['betawidth'])            
            if np.count_nonzero(betas) == 0:
                print('Warning, all betas were 0!')
                print(param_comb)
                sys.exit(0)
            param_comb['sigma'] = sigma
            param_comb['betas'] = betas
            # Save a seed that will be used to generate the same data for every process
            param_comb['seed'] = i 
        with open(arg_file, 'wb') as f:
            # Sequentially pickle the elements of iter_param_list so they can be 
            # sequentially unpickled
            
            # First pickle away the number of tasks 
            f.write(pickle.dumps(len(iter_param_list)))
            # Then the number of features (assuming it remains fixed)
            f.write(pickle.dumps(param_comb['n_features']))
                
            for elem in iter_param_list:
                f.write(pickle.dumps(elem))
            
        print('arg_file iteration time: %f' % (time.time() - start))
    
    return paths, ntasks

# Store the metadata as an easily searchable pandas dataframe
def generate_log_file(argfile_array, jobdir):
    metadata = pd.DataFrame(argfile_array)
    metadata.to_pickle('%s/log.dat' % jobdir)
    
def generate_sbatch_scripts(sbatch_array, sbatch_dir, script_dir):

    # Generate sbatch scripts for the given directory

    for i, sbatch in enumerate(sbatch_array):
        
        sbname = '%s/sbatch%d.sh' % (sbatch_dir, i)
        jobname = '%s_job%d' % (sbatch['exp_type'], i)

        script = 'mpi_submit.py'
        results_file = '%s/%s.dat' % (sbatch_dir, jobname)

        with open(sbname, 'w') as sb:
            # Arguments common across jobs
            sb.write('#!/bin/bash\n')
            sb.write('#SBATCH -q regular\n')
            sb.write('#SBATCH -N 1\n')

            sb.write('#SBATCH -t %s\n' % sbatch['job_time'])

            sb.write('#SBATCH --job-name=%s\n' % jobname)
            sb.write('#SBATCH --out=%s/%s.o\n' % (sbatch_dir, jobname))
            sb.write('#SBATCH --error=%s/%s.e\n' % (sbatch_dir, jobname))
            sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
            sb.write('#SBATCH --mail-type=FAIL\n')

            # Work with out own Anaconda environment
            # To make this work, we had to add some paths to our .bash_profile.ext
            sb.write('source ~/anaconda3/bin/activate\n')
            sb.write('source activate nse\n')
            
            # Critical to prevent threads competing for resources
            sb.write('export OMP_NUM_THREADS=1\n')
            sb.write('export MKL_NUM_THREADS=1\n')
            sb.write('export KMP_AFFINITY=disabled\n')

            sb.write('srun -n %d python3 -u %s/%s %s %s %s' 
                    % (sbatch['ntasks'], script_dir, script, sbatch['arg_file'],
                       results_file, sbatch['exp_type']))

# Use skip_argfiles if arg_files have already been generated and just need to 
# re-gen sbatch files
def create_job_structure(submit_file, jobdir, skip_argfiles = False, single_test = False):

    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    path, name = submit_file.split('/')
    sys.path.append(path)
    args = importlib.import_module(name)

    iter_params = args.iter_params
    comm_params = args.comm_params
    exp_types = args.exp_types
    algorithm_times = args.algorithm_times
    script_dir = args.script_dir

    if not skip_argfiles:
        # Common master list of arg file parameters
        argfile_array = []

        iter_keys = list(iter_params.keys())

        # Iterate over all combinations of parameters in iter_params and combine them
        # with comm_params to produce a unique argument dictionary for each individual job
        for arg_comb in itertools.product(*list(iter_params.values())):
            arg = {}
            for j in range(len(arg_comb)):
                arg[iter_keys[j]] = arg_comb[j]         
            for key, value in comm_params.items():
                arg[key] = value

            argfile_array.append(arg)
            # Create only a single file for testing purposes
            if single_test:
                break

        # Generate log file
        generate_log_file(argfile_array, jobdir)

        # Master directory with all arg files
        if not os.path.exists('%s/master' % jobdir):
            os.mkdir('%s/master' % jobdir)

        # Generate the arg_files:
        paths, ntasks = generate_arg_files(argfile_array, jobdir)

    else:
        # Need to get paths and ntasks
        paths = glob('%s/master/*.dat' % jobdir)
        # Go through and count the length of the dictionary contained within each argfile
        # to get ntasks
        ntasks = []
        for path in paths:
            with open(path, 'rb') as f:
                ntasks.append(pickle.load(f))
        
    # Create an sbatch_array used to generate sbatch files that specifies exp_type, job_time, 
    # num_tasks, and the path to the corresponding arg_file

    sbatch_array = []
    for i, exp_type in enumerate(exp_types):
        sbatch_array.append([])
        for j in range(len(paths)):
            sbatch_dict = {
            'arg_file' : paths[j],
            'ntasks' : min(24, ntasks[j]),
            'exp_type' : exp_type,
            'job_time' : algorithm_times[i]
            }
            sbatch_array[i].append(sbatch_dict)

    # Generate the directory structure and sbatch files

    for i, exp_type in enumerate(exp_types):
        
        if not os.path.exists('%s/%s' % (jobdir, exp_type)):
            os.mkdir('%s/%s' % (jobdir, exp_type))

        generate_sbatch_scripts(sbatch_array[i], '%s/%s' % (jobdir, exp_type),
                               script_dir)        

# Jobdir: Directory to crawl through
# size: only submit this many jobs (if exp_type specified, this
# means only submit this many jobs of this exp_type(s))
# exp_type: Only run these kinds of experiments
# edit_attribute: Dict containing key value pair of job property to
# edit before submitting
# run: If set to false, make modifications to sbatch files but do not run them
def run_jobs(jobdir, constraint, size = None, nums = None,
             exp_type = None, run = False):
    
    # Crawl through all subdirectories and 
    # (1) change permissions of sbatch file
    # (2) run sbatch file

    run_files = grab_sbatch_files(jobdir, exp_type)
    
    # Can either constrain size or manually give numbers
    if size is not None:
        run_files = run_files[:size]
    elif nums is not None:
        run_files = [r for r in run_files if int(r.split('params')[1].split('.dat')[0]) in nums]
    cont = input("You are about to submit %d jobs, do you want to continue? [0/1]" % len(run_files))
    
    if cont:
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
                # Change permissions
                os.system('chmod u+x %s' % run_file)
                # Submit the job
                os.system('sbatch %s ' % run_file)

# Sequentially run files locally:
def run_jobs_local(jobdir, nprocs, size = None, exp_type = None):
    # Crawl through all subdirectories and 
    # (1) change permissions of sbatch file
    # (2) run sbatch file
    run_files = grab_arg_files(jobdir, exp_type)
            
    if size is not None:
        run_files = run_files[:size]
    cont = input("You are about to submit %d jobs, do you want to continue? [0/1]" % len(run_files))

    if cont:
        for run_file in run_files:
            msg = check_output('mpiexec -n %d python -u mpi_submit.py %s' 
                          % (nprocs, run_file))        
            print(msg) 
# Edit specific lines of all sbatch files 
# By default, edits an attribute
# To replace a specific line with exact match to string, set edit_attribute
# to None, and pass in linestring instead
def edit_job_attribute(jobdir, edit_attribute, linestring = None, exp_type = None):
    
    run_files = grab_sbatch_files(jobdir, exp_type)

    for run_file in run_files:
        start = time.time()
        f = open(run_file, 'r')
        contents = f.readlines()
        f.close()

        if edit_attribute is not None:
            for key, value in edit_attribute.items():
                key_string = [s for s in contents if ' %s' % key in s][0]
                key_string_idx = contents.index(key_string)
                if '--' in key:
                    new_key_string = '#SBATCH %s=%s\n' % (key, value)
                else:
                    new_key_string = '#SBATCH %s %s\n' % (key, value)
                contents.remove(key_string)
                contents.insert(key_string_idx, new_key_string)
        elif linestring is not None:
            for key, value in linestring.items():
                key_string = [s for s in contents if key in s][0]
                key_string_idx = contents.index(key_string)
                contents.remove(key_string)
                contents.insert(key_string_idx, value)

        f = open(run_file, "w")
        contents = "".join(contents)
        f.write(contents)
        f.close()
        print('Iteration time: %f' % (time.time() - start))
        
# Crawl through a subdirectory and grab all jobs contained in it:
def grab_sbatch_files(root_dir, exp_type = None):
    run_files = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if exp_type is not None:
                if exp_type in p:
                    run_files.extend(glob('%s/*.sh' % p))
            else:
                run_files.extend(glob('%s/*.sh' % p))
    return run_files
    
# Upon running this command, check which jobs have been successfully completed, and 
# submit another batch of jobs to be run
def evaluate_job_structure(root_dir):

    # Load log:
    with open('%s/log.dat' % root_dir) as f:
        pickle.load(f)

    # For each job directory, the directory has either not been submitted, 
    # only partially completed, or completely completed

    # If not submitted, mark for submission

    # Also keep track of jobs currently running (WARNING this will not account for jobs that
    # are running on )
    currently_running = np.zeros(run_status.shape)

    # If only partially complete, record indices of jobs that must be re-submitted
    for i in range(len(job_array_chunks)):
        for j in range(len(job_array_chunks[i])):
            jobdir = job_array_chunks[i][j][0]['jobdir']
            # Get all output files in directory:
            out_files = glob.glob('%s/*.o')
            if out_files == []:
                continue
            else:
                check_outout(run_status[i, j, :], completion_status[i, j, :], out_files)

    ## TO-DO: mark jobs that are currently running

    # Everytime this script is called, submit an additional 1000 jobs to the queue

    # In order of priority:
    # Directories that have not yet been run
    # Directories that have been run, are not currently running, and have jobs that
    # were not successfully completed
    
        

# # Read through all the out files 
# # and determine whether a given job
# # has finished successfully
# def check_output(rstatus, cstatus, out_files):
    
#     for i in range(rstatus.size):
#         # Trust that if the job has been marked completed, it has been:
#         if cstatus[i]:
#             continue
#         # If the job has run, determine whether or not it has been completed
#         if rstatus[i]:
#             if 'job%d.o' % i in out_files:
#                 # Check the last line for completed message
#                 with open('job%d.o' % i, 'r') as f:
#                     first = f.readline()        # Read the first line.
#                     f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
#                     while f.read(1) != b"\n":   # Until EOL is found...
#                         f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
#                     last = f.readline()         # Read last line.
#                 if 'Job completed!' in last:
#                     cstatus[i] = 1
