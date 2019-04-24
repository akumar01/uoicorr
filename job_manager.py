import numpy as np
import importlib
import itertools
import pdb
import sys, os
import pickle
import json
import time
import traceback
from glob import glob
from subprocess import check_output

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

def generate_arg_files(job_array, results_files, jobdir):

    for i, arg in enumerate(job_array):
        start = time.time()
        arg['results_file'] = results_files[i]

        jobname = '%s_%s_job%d' % (arg['exp_type'], arg['chunk_id'], i)

        arg_file = '%s/%s_params.dat' % (jobdir, jobname)

#         for key, value in arg.items():
#             arg[key] = fix_datatypes(value)
             
        with open(arg_file, 'wb') as f:
            try:
                f.write(pickle.dumps(arg, pickle.HIGHEST_PROTOCOL))
            except:
                traceback.print_exc()
                pdb.set_trace()

        arg['arg_file'] = arg_file
        arg['jobdir'] = jobdir

        job_array[i] = arg
        print('arg_file iteration time: %f' % (time.time() - start))
    
    return job_array

def generate_log_file(job_array, jobdir):

    # Collect the unqiue values comprised in each job dictionary
    values = list(job_array[0].keys())
    unique_args = []
    for i, value in enumerate(values):
        pdb.set_trace()
        unique_args.append(np.unique(np.array([jbarr[value] for jbarr in job_array])))
    
    # Write to file
    with open('%s/log.txt' % jobdir, 'w') as f:
        for i, value in enumerate(values):
            f.write('%s: %s\n' % (value, np.array2string(unique_args[i])))

def generate_sbatch_scripts(job_array, script_dir):

    # Generate sbatch scripts for the given directory

    for i, job in enumerate(job_array):
        
        jobdir = job['jobdir']
        sbname = '%s/sbatch%d.sh' % (jobdir, i)
        jobname = '%s_%s_job%d' % (job['exp_type'], job['chunk_id'], i)

        script = 'mpi_submit.py'

        # If UoI, parallelize across bootstraps
        # Otherwise, take the max of 64 and the number of reps required
        # by this particular job
        if job['exp_type'] in ['UoIElasticNet', 'UoILasso']:
            nprocs = 48
        else:
            nprocs = 64

        with open(sbname, 'w') as sb:
            # Arguments common across jobs
            sb.write('#!/bin/bash\n')
            sb.write('#SBATCH -q regular\n')
            sb.write('#SBATCH -N 1\n')

            sb.write('#SBATCH -t %s\n' % job['job_time'])

            sb.write('#SBATCH --job-name=%s\n' % jobname)
            sb.write('#SBATCH --out=%s/%s.o\n' % (jobdir, jobname))
            sb.write('#SBATCH --error=%s/%s.e\n' % (jobdir, jobname))
            sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
            sb.write('#SBATCH --mail-type=FAIL\n')

            # Work with out own Anaconda environment
            # To make this work, we had to add some paths to our .bash_profile.ext
            sb.write('source ~/anaconda3/bin/activate\n')
            sb.write('source activate nse\n')
            
            # Critical to prevent threads competing for resources
            sb.write('export OMP_NUM_THREADS=1\n')
            sb.write('export MKL_NUM_THREADS=1\n')

            sb.write('srun -n %d python3 -u %s/%s %s' 
                    % (nprocs, script_dir, script, job['arg_file']))

def create_job_structure(arg_file, data_dir = 'uoicorr/dense', skip_arg_files=True):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    path, name = arg_file.split('/')
    sys.path.append(path)
    args = importlib.import_module(name)

    iter_params = args.iter_params
    comm_params = args.comm_params
    exp_types = args.exp_types
    algorithm_times = args.algorithm_times
    script_dir = args.script_dir
    total_jobs = np.prod(len(val) for val in iter_params.values())

    # Master list of job parameters
    job_array = []

    iter_keys = list(iter_params.keys())

    for i, exp_type in enumerate(exp_types):
        job_array.append([])
        # Iterate over all combinations of parameters in iter_params and combine them
        # with comm_params to produce a unique argument dictionary for each individual job
        for j, arg_comb in enumerate(itertools.product(*list(iter_params.values()))):
            arg = {}
            for j in range(len(arg_comb)):
                arg[iter_keys[j]] = arg_comb[j]         
            for key, value in comm_params.items():
                arg[key] = value
            arg['exp_type'] = exp_type

            job_array[i].append(arg)

    # Separate into sets of 1000 for each exp_type
    pdb.set_trace()
    chunk_size = 1000
    n_chunks = int(len(job_array[0])/chunk_size)

    job_array_chunks = []
    for i, exp_type in enumerate(exp_types):
        job_array_chunks.append(list(chunk_list(job_array[i], chunk_size)))

    # Generate the directory structure
    for i, exp_type in enumerate(exp_types):
        
        if not os.path.exists('%s/%s' % (data_dir, exp_type)):
            os.mkdir('%s/%s' % (data_dir, exp_type))

        # For each exp_type directory, generate a subdiretory for 
        # each chunk of jobs
        for j in range(len(job_array_chunks[i])):

            # Set the job time according to the worst-case runtime of the particular algorithm
            for job in job_array_chunks[i][j]:
                job['job_time'] = algorithm_times[i]
                job['chunk_id'] = j

            jobdir = '%s/%s/%s' % (data_dir, exp_type, j)
            if not os.path.exists(jobdir):
                os.mkdir(jobdir)
            # Generate a human-readable log file, sbatch 
            # script, and job param files for this directory

            results_files = ['%s/job%d.h5' % (jobdir, i) for i in range(len(job_array_chunks[i][j]))]

#            generate_log_file(job_array_chunks[i][j], jobdir)
            job_array_chunks[i][j] = generate_arg_files(job_array_chunks[i][j], results_files, jobdir)
            generate_sbatch_scripts(job_array_chunks[i][j], script_dir)
        

#     # Initialize arrays that keep track of whether particular jobs have run and whether they have
#     # succesfully completed            
#     run_status = np.zeros((len(job_array_chunks), len(job_array_chunks[0]), len(job_array_chunks[0][0])))
#     completion_status = np.zeros(run_status.shape)


#     # Store away:
#     with open('%s/log.dat' % data_dir, 'wb') as f:
#         pickle.dump(f, [job_array_chunks, run_status, completion_status])

# Jobdir: Directory to crawl through
# size: only submit this many jobs (if exp_type specified, this
# means only submit this many jobs of this exp_type(s))
# exp_type: Only run these kinds of experiments
# edit_attribute: Dict containing key value pair of job property to
# edit before submitting
# run: If set to false, make modifications to sbatch files but do not run them
def run_jobs(jobdir, constraint, size = None, exp_type = None, run = True):
    
    # Crawl through all subdirectories and 
    # (1) change permissions of sbatch file
    # (2) run sbatch file

    run_files = grab_sbatch_files(jobdir, exp_type)
            
    if size is not None:
        run_files = run_files[:size]
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
            
            # Add constraint to the top
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
    
# Crawl through a subdirectory and grab all param files
def grab_arg_files(root_dir, exp_type = None):
    run_files = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if exp_type is not None:
                if exp_type in p:
                    run_files.extend(glob('%s/*.dat' % p))
            else:
                run_files.extend(glob('%s/*.dat' % p))
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
