import numpy as np
import itertools
import pdb
import os
import pickle
import json

def chunk_list(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def generate_arg_files(job_array, results_files, jobdir):

    for i, arg in enumerate(job_array):

        arg['results_file'] = results_files[i]

        jobname = '%s_%s_job%d' % (arg['exp_type'], arg['chunk_id'], i)

        arg_file = '%s/%s_params.json' % (jobdir, jobname)

        # Numpy datatypes are not JSON-serializable
        for key, value in arg.items():
            if type(value) == np.int_:
                arg[key] = int(value)
            if type(value) == np.float_:
                arg[key] = float(value)

        with open(arg_file, 'w') as f:
            json.dump(arg, f)
            f.close()

        arg['arg_file'] = arg_file
        arg['jobdir'] = jobdir
    return job_array

def generate_log_file(job_array, jobdir):

    # Collect the unqiue values comprised in each job dictionary
    values = list(job_array[0].keys())
    unique_args = []
    for i, value in enumerate(values):
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

        # Use MPI to accelerate UoI
        if job['exp_type'] in ['UoILasso' or 'UoIElasticNet']:
            script = 'mpi_submit.py'
        else:
            script = 'uoicorr_base.py'

        with open(sbname, 'w') as sb:
            # Arguments common across jobs
            sb.write('#!/bin/bash\n')
            sb.write('#SBATCH -q shared\n')
            sb.write('#SBATCH -n 1\n')
            sb.write('#SBATCH -t %s\n' % job['job_time'])

            sb.write('#SBATCH --job-name=%s\n' % jobname)
            sb.write('#SBATCH --out=%s/%s.o\n' % (jobdir, jobname))
            sb.write('#SBATCH --error=%s/%s.e\n' % (jobdir, jobname))
            sb.write('#SBATCH --mail-user=ankit_kumar@berkeley.edu\n')
            sb.write('#SBATCH --mail-type=FAIL\n')

            # Work with out own Anaconda environment
            # To make this work, we had to add some paths to our .bash_profile.ext
            sb.write('source activate nse\n')

            # Specify architecture 
            sb.write('if [[hostname == *"cori"* ]]; then\n')
            sb.write('  #SBATCH -C haswell\n')       
            sb.write('fi\n') 
            sb.write('srun python3  %s/%s %s' 
                    % (script_dir, script, job['arg_file']))

def create_job_structure(data_dir = 'uoicorr/dense'):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    script_dir = '/global/homes/a/akumar25/repos/uoicorr'

    ###### Master list of parameters to be iterated over #######

    exp_types =  ['UoILasso', 'UoIElasticNet', 'EN', 'CV_Lasso']

    # Upon each run, the status of completed jobs will be compared
    # to that required by this list.

    iter_params = {

    'betawidth' : [0, np.inf],

    # Sparsity
    'sparsity' : np.logspace(0.01, 1, 20),

    # Linear interpolation strength
    'interp_t': np.arange(0, 11, 10),

    # Block sizes
    'block_sizes' : [5, 10, 20, 50, 100],

    # Block correlation
    'correlation' : np.linspace(0, 0.8, 10),

    # Exponential length scales
    'L' : [1, 2.5, 5, 7.5, 10],

    # n/p ratio #
    'np_ratio': [0.5, 1, 2, 3, 5],

    }

    #############################################################

    ##### Common parameters held fixed across all jobs ##########
    comm_params = {
    'reg_params': [],
    'n_models': 1,
    'kappa' : 0.3, 
    'n_features' : 500,
    'est_score': 'r2',
    'reps' : 10,
    'stability_selection' : [1.0],
    'n_boots_sel': 48,
    'n_boots_est' : 48}

    # Parameters for ElasticNet
    comm_params['l1_ratios'] = [0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
    comm_params['n_alphas'] = 48

    ###############################################################

    total_jobs = np.prod(len(val) for val in iter_params.values())

    # Estimated worst case run-time for a single repitition for each algorithm in exp_types 
    algorithm_times = ['10:00:00', '10:00:00', '10:00:00', '10:00:00']

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

            results_files = ['job%d.h5' % i for i in range(len(job_array_chunks[i][j]))]

            generate_log_file(job_array_chunks[i][j], jobdir)
            job_array_chunks[i][j] = generate_arg_files(job_array_chunks[i][j], results_files, jobdir)
            generate_sbatch_scripts(job_array_chunks[i][j], script_dir)

    # Initialize arrays that keep track of whether particular jobs have run and whether they have
    # succesfully completed            
    run_status = np.zeros((len(job_array_chunks), len(job_array_chunks[0]), len(job_array_chunks[0][0])))
    completion_status = np.zeros(run_status.shape)

    # Store away:
    with open('%s/log.dat' % data_dir, 'wb') as f:
        pickle.dump(f, [job_array_chunks, run_status, completion_status])

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
    
        

# Read through all the out files 
# and determine whether a given job
# has finished successfully
def check_output(rstatus, cstatus, out_files):

    for i in range(rstatus.size):
        # Trust that if the job has been marked completed, it has been:
        if cstatus[i]:
            continue
        # If the job has run, determine whether or not it has been completed
        if rstatus[i]:
            if 'job%d.o' % i in out_files:
                # Check the last line for completed message
                with open('job%d.o' % i, 'r') as f:
                    first = f.readline()        # Read the first line.
                    f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
                    while f.read(1) != b"\n":   # Until EOL is found...
                        f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
                    last = f.readline()         # Read last line.
                if 'Job completed!' in last:
                    cstatus[i] = 1