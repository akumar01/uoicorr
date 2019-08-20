import numpy as np
import importlib
import re
import itertools
import pdb
import sys, os
import pickle
import struct
import time
import traceback
from shlex import split
from copy import deepcopy
import pandas as pd
from glob import glob
from subprocess import check_output
import subprocess
from utils import gen_covariance, gen_beta2, gen_data
from misc import group_dictionaries, unique_obj

from job_utils.idxpckl import Indexed_Pickle

# Take two integers and generate a unique, single integer from them
def gen_seed(i, j, dimi, dimj):
    # i_list = list(map(int, str(i)))
    # j_list = list(map(int, str(j)))

    # # Use the method of interleaving
    # k_list = list(itertools.chain.from_iterable(
    #               itertools.zip_longest(i_list, j_list)))

    # # Filter out Nones
    # k_list = [k for k in k_list if k is not None]

    # # Return to the a number
    # seed = int("".join(map(str, k_list)))

    seed = np.ravel_multi_index((i, j), dims=(dimi, dimj))

    return seed

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
    # Keep track of seeds so that they are not repeated
    seeds = []
    for j, arg_ in enumerate(argfile_array):
        start = time.time()

        # Generate the full set of data/metadata required to run the job
        
        sub_iter_params = arg_['sub_iter_params']
        if sub_iter_params: 
            for key in sub_iter_params:
                if not hasattr(arg_[key], '__len__'):
                    arg_[key] = [arg_[key]]
                arg_[key] = list(arg_[key])

        
        # Complement of the sub_iter_params:
        const_keys = list(set(arg_.keys()) - set(sub_iter_params))

        const_args = {k: arg_[k] for k in const_keys}
        # Make sure the keys match for selection methods and fields
        const_args['selection_methods'].sort()
        field_keys = list(const_args['fields'].keys())
        field_keys.sort()

        assert(const_args['selection_methods'] == field_keys)

        arg_comb = list(itertools.product(*[arg_[key] for key in arg_['sub_iter_params']]))

        iter_param_list = []
        for i in range(len(arg_comb)):
            arg_dict = const_args.copy()
            [arg_dict.update({arg_['sub_iter_params'][j]: arg_comb[i][j] for j in range(len(arg_['sub_iter_params']))})]
            iter_param_list.append(arg_dict)

        for i, param_comb in enumerate(iter_param_list):
            # Assign each param combination a unique index before multiplying by reps
            # so we can easily group identical combinations together later
            param_comb['index'] = i
            if 'n_samples' in list(param_comb.keys()):
                n_samples = param_comb['n_samples']
            elif 'np_ratio' in list(param_comb.keys()):
                n_samples = int(param_comb['np_ratio'] * param_comb['n_features'])
                param_comb['n_samples'] = n_samples

        # Make n_reps independent copies of iter_param_list
        iter_param_list = [deepcopy(iter_param_list) for _ in range(arg_['reps'])]
        iter_param_list = [elem for sublist in iter_param_list for elem in sublist]

        # Seed AFTER copying for reps
        for i, param_comb in enumerate(iter_param_list):
            seed = gen_seed(i, j, len(iter_param_list), len(argfile_array))
            param_comb['seed'] = seed
            seeds.append(seed)
            
        ntasks.append(len(iter_param_list))
        arg_file = '%s/master/params%d.dat' % (jobdir, j)
        paths.append(arg_file)

        f = Indexed_Pickle(arg_file)
        f.init_save(len(iter_param_list), header_data = 
                    {'n_features' : param_comb['n_features'], 
                     'selection_methods' : param_comb['selection_methods'],
                     'fields' : param_comb['fields']})
        for elem in iter_param_list:
            f.save(elem)

        f.close_save()
        print('arg_file iteration time: %f' % (time.time() - start))

    # Check that all random number seeds are unique
    assert(len(np.unique(seeds)) == len(seeds))    

    return paths, ntasks

# Store the metadata as an easily searchable pandas dataframe
def generate_log_file(argfile_array, jobdir, desc = None):
    metadata = pd.DataFrame(argfile_array)
    # Description of simulation
    metadata.desc = desc
    metadata.to_pickle('%s/log.dat' % jobdir)

    
def generate_sbatch_scripts(sbatch_array, sbatch_dir, script_dir):

    # Generate sbatch scripts for the given directory

    # Quality of service: If running on shared queue, then do not
    # set up MPI parameters and request only a single core
    
    for i, sbatch in enumerate(sbatch_array):
        qos = sbatch['qos']    
       
        if 'sbname' not in list(sbatch.keys()):
            sbname = 'sbatch%d.sh' % i
        else:
            sbname = sbatch['sbname']
            
        sbname = '%s/%s' % (sbatch_dir, sbname)

        if 'jobname' not in list(sbatch.keys()):
            jobname = '%s_job%d' % (sbatch['exp_type'], i)
        else:
            jobname = sbatch['jobname']

        script = 'mpi_submit.py'
        results_file = '%s/%s' % (sbatch_dir, jobname)

        with open(sbname, 'w') as sb:
            # Arguments common across jobs
            sb.write('#!/bin/bash\n')
            if qos == 'regular':
                sb.write('#SBATCH --qos=regular\n')
                sb.write('#SBATCH -N 1\n')
            else:
                sb.write('#SBATCH --qos=shared\n')
            
            sb.write('#SBATCH -n %d\n' % sbatch['ntasks'])
            sb.write('#SBATCH -c %d\n' % sbatch['cpt'])    
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
            
            if qos == 'regular':
                # Critical to prevent threads competing for resources
                sb.write('export OMP_NUM_THREADS=1\n')
                sb.write('export KMP_AFFINITY=disabled\n')
            
            # Broadcast script to compute nodes
            # sb.write('sbcast -f --compress %s/%s /tmp/%s\n' % (script_dir, script, script))

            # sb.write('sbcast -f --compress %s/%s /tmp/%s\n' % (script_dir, script, script))
            sb.write('srun python3 -u -m mpi4py %s/%s %s %s %s' 
                     % (script_dir, script, sbatch['arg_file'],
                     results_file, sbatch['exp_type']))
                
# Use skip_argfiles if arg_files have already been generated and just need to 
# re-gen sbatch files
    
# srun_opts: options to feed into the srun command (for example, n tasks, n cpus per task)
def create_job_structure(submit_file, jobdir, qos, numtasks, cpu_per_task,
                         skip_argfiles = False, single_test = False): 
                         
    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    path, name = submit_file.split('/')
    name = os.path.splitext(name)[0]
    sys.path.append(path)
    args = importlib.import_module(name)

    # Copy submit file to jobdir
    os.system('cp %s %s/' % (submit_file, jobdir))

    iter_params = args.iter_params
    comm_params = args.comm_params
    exp_types = args.exp_types
    algorithm_times = args.algorithm_times
    script_dir = args.script_dir

    if hasattr(args, 'desc'):
        desc = args.desc
    else:
        desc = 'No description available.'

    if not skip_argfiles:
        # Common master list of arg file parameters
        argfile_array = []

        if iter_params:
        
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
        else: 
            # No iter params (generate just a single arg file)
            argfile_array = [comm_params]

        # Generate log file
        generate_log_file(argfile_array, jobdir, desc)

        # Master directory with all arg files
        if not os.path.exists('%s/master' % jobdir):
            os.mkdir('%s/master' % jobdir)
        # Generate the arg_files:
        paths, ntasks = generate_arg_files(argfile_array, jobdir)

    else:
        # Need to get paths and ntasks
        paths = glob('%s/master/*.dat' % jobdir)
        
    # Create an sbatch_array used to generate sbatch files that specifies exp_type, job_time, 
    # num_tasks, and the path to the corresponding arg_file

    sbatch_array = []
    for i, exp_type in enumerate(exp_types):
        sbatch_array.append([])
        for j in range(len(paths)):
            sbatch_dict = {
            'arg_file' : paths[j],
            'ntasks' : numtasks,
            'qos' : qos,
            'cpt' : cpu_per_task,
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
def run_jobs(jobdir, constraint, size = None, nums = None, run_files = None,
             exp_type = None, run = False):
    
    # Crawl through all subdirectories and 
    # (1) change permissions of sbatch file
    # (2) run sbatch file
    if run_files is None:
        run_files = grab_files(jobdir, '*.sh', exp_type)
    
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
            
            # Add constraint after declaration of qos
            if constraint == 'haswell':
                contents.insert(2, '#SBATCH --constraint=haswell\n')
              
            if constraint == 'knl':
                contents.insert(2, '#SBATCH --constraint=knl\n')                  
            
            f = open(run_file, "w")
            contents = "".join(contents)
            f.write(contents)
            f.close()
            
            if run:
                run_(run_file)
                
# Find jobs that are lacking a .h5 file (jobs that failed to run)
def unfinished_jobs(jobdir, exp_type = None):
    
    # Get all potential files to run
    all_files = grab_files(jobdir, '*.sh', exp_type)
    # Get all files with a .h5 output
    completed_files = grab_files(jobdir, '*.dat', exp_type)            
                
    # Get the job numbers to compare 
    all_files = [os.path.split(f)[1] for f in all_files]
    all_jobnos = [int(f.split('.sh')[0].split('sbatch')[1]) for f in all_files]
    
    completed_files = [os.path.split(f)[1] for f in completed_files]
    completed_jobnos = [int(f.split('.dat')[0].split('job')[1]) for f in completed_files]
    
    to_run = np.setdiff1d(all_jobnos, completed_jobnos)
    
    # Reconstruct the full paths from the numbers
    run_paths = paths_from_nums(jobdir, exp_type, to_run, 'sbatch')
    return run_paths

# Run the given jobs:
def run_(run_file):
    os.system('chmod u+x %s' % run_file)
    os.system('sbatch %s' % run_file)
                
# Sequentially run files locally:
def run_jobs_local(jobdir, nprocs, run_files = None, size = None, exp_type = None,
                   script_dir=None, background = False, reverse=False):
    # Crawl through all subdirectories and 
    # (1) Grab the sbatch files
    # (2) Extract the srun statement
    # (3) Replace srun with mpiexec 
    # (4) Replace script and data dirs with local machine paths
    # (5) Run 

    # Reverse: start from the end of run_files list if size is True

    if run_files is None:
        run_files = grab_files(jobdir, '*.sh', exp_type)

    if size is not None:
        if reverse:
            run_files = run_files[-size:]
        else:
            run_files = run_files[:size]
    cont = input("You are about to submit %d jobs, do you want to continue? [0/1]" % len(run_files))

    if cont:
        for run_file in run_files:
            with open(run_file, 'r') as f:
                fcontents = f.readlines()

            srun_string = [s for s in fcontents if 'srun' in s][0]

            # replace srun with mpiexec -nprocs
            mpi_string = srun_string.replace('srun', 'mpiexec -n %d' % nprocs)
            mpi_string = split(mpi_string)

            # get rid of the -m mpi4py flags
            mpi_string.pop(5)
            mpi_string.pop(5)

            # Replace script and data dirs with local machine paths
            if script_dir is not None:
                mpi_string[5] = script_dir

            # Replace the data path with the local machine path, if they are 
            # not the same
            run_file_root_path = '/'.join(run_file.split('/')[:-2])
            mpi_string_suffix = '/'.join(mpi_string[6].split('/')[-2:])
            mpi_string[6] = run_file_root_path + '/%s' % mpi_string_suffix

            mpi_string_suffix = '/'.join(mpi_string[7].split('/')[-2:])
            mpi_string[7] = run_file_root_path + '/%s' % mpi_string_suffix

            # Spawn process in the background of shell, only do this if you 
            # trust the program will work independently
            if background:
                subprocess.Popen(mpi_string, shell = True)
            else:
                for output in local_exec(mpi_string):
                    print(output)

# Copied from this stackoverflow post: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
# Capture stdout in real time 
def local_exec(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

# Edit specific lines of the provided sbatch files (full paths)
# By default, edits an attribute
# To replace a specific line with exact match to string, set edit_attribute
# edit attirubte must be dictionary key pairs
# to None, and pass in linestring instead
def set_job_attribute(run_files, edit_attribute, linestring = None, exp_type = None):
    
    bad_idxs = np.zeros(len(run_files))

    for ridx, run_file in enumerate(run_files):
        start = time.time()
        f = open(run_file, 'r')
        contents = f.readlines()
        f.close()

        if edit_attribute is not None:
            for key, value in edit_attribute.items():
                try:
                    key_string = [s for s in contents if ' %s' % key in s][0]
                    key_string_idx = contents.index(key_string)
                    if '--' in key:
                        new_key_string = '#SBATCH %s=%s\n' % (key, value)
                    else:
                        new_key_string = '#SBATCH %s %s\n' % (key, value)
                    contents.remove(key_string)
                    contents.insert(key_string_idx, new_key_string)
                except:
                    bad_idxs[ridx] = 1
 
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

    return bad_idxs

# Open up an sbatch file and grab an attributes like job time that cannot be read off from 
# anywhere else
def get_job_attribute(run_files, attribute, exp_type = None):

    attribute_vals = []

    for run_file in run_files:  
        with open(run_file, 'r') as f:
            contents = f.readlines()
        
        attribute_string = [s for s in contents if ' %s' % attribute in s][0]
        if '--' in attribute:
            attribute_value = attribute_string.split('%s=' % attribute)[1].split('\n')[0]            
        elif '-' in attribute:
            attribute_value = attribute_string.split('%s ' % attribute)[1].split('\n')[0]
        attribute_vals.append(attribute_value)

    return attribute_vals
 
# Remove line
def remove_line(run_files, index):
    
    for run_file in run_files:
        
        f = open(run_file, 'r')
        contents = f.readlines()
        f.close()
        contents.pop(index)

        f = open(run_file, 'w')
        contents = "".join(contents)
        f.write(contents)
        f.close()
        
# Change the arguments sent into the srun argument
def edit_srun_statement(run_files, srun_args):

    for run_file in run_files:

        f = open(run_file, 'r')
        contents = f.readlines()
        f.close()

        srun_string = [s for s in contents if 'srun' in s][0]
        srun_string_idx = contents.index(srun_string)
        srun_split = srun_string.split('python')

        new_srun = 'srun %s ' % srun_args

        srun_split[0] = new_srun
        new_srun_string = "".join(srun_string)        
        contents.insert(srun_string_idx, new_srun_string)

        f = open(run_file, 'w')
        contents = "".join(contents)
        f.write(contents)
        f.close()

# Crawl through a subdirectory and grab all files contained in it matching the 
# provided criteria:
def grab_files(root_dir, file_str, exp_type = None):
    run_files = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if exp_type is not None:
                if exp_type in p:
                    run_files.extend(glob('%s/%s' % (p, file_str)))
            else:
                run_files.extend(glob('%s/%s' % (p, file_str)))


    #run_files = natsort.natsorted(run_files)
    return run_files

# Return the full path to the file type given a list of jobIDs
def paths_from_nums(jobdir, exp_type, nums, type_):
    
    base_path = '%s/%s' % (jobdir, exp_type)
    paths = []
    for n in nums:
        if type_ == 'sbatch':
            paths.append('%s/sbatch%d.sh' % (base_path, n))
        elif type == 'data':
            paths.append('%s/%s_job%d.dat' % (base_path, exp_type, n))
        elif type == 'output':
            paths.append('%s/%s_job%d.o' % (base_path, exp_type, n))
        elif type == 'error':
            paths.append('%s/%s_job%d.e' % (base_path, exp_type, n))
    return paths


# Split the jobs given by jobnums into nsplits.
def split_job(jobdir, exp_type, jobnums, n_splits):

    # Grab the param files and sbatch files
    param_files = []
    for j in jobnums:
        param_files.extend(grab_files(jobdir, 'params%d.dat' % j, 'master'))

    sbatch_files = []
    for j in jobnums:
        sbatch_files.extend(grab_files(jobdir, 'sbatch%d.sh' % j, exp_type))

    # Load the param files
    for i, param_file in enumerate(param_files):
        f = open(param_file, 'rb')
        index_loc = f.read(8)
        index_loc = struct.unpack('L', index_loc)[0]
        total_tasks = pickle.load(f)
        n_features = pickle.load(f)
        f.seek(index_loc, 0)
        index = pickle.load(f)

        # Distribute parameters across multiple files
        task_splits = np.array_split(np.arange(total_tasks), n_splits)
        
        split_param_files = ['%s_split%d.dat' % (param_file.split('.dat')[0], ii) for ii in range(n_splits)]

        for j, split_param_file in enumerate(split_param_files):
            with open(split_param_file, 'wb') as f2:
                
                f2.write(struct.pack('L', 0))     
                f2.write(pickle.dumps(len(task_splits[j])))
                f2.write(pickle.dumps(n_features))

                split_index = []
                # Load the needed param dict from f:
                for task in task_splits[j]:
                    f.seek(index[task], 0)
                    params = pickle.load(f)
                    split_index.append(f2.tell())
                    f2.write(pickle.dumps(params))
                split_index_loc = f2.tell()
                f2.write(pickle.dumps(split_index))
                f2.seek(0, 0)
                f2.write(struct.pack('L', split_index_loc))

        f.seek(0, 0)
        f.close()

        # Create corresponding sbatch files
        split_sbatch_files = ['%s_split%d.sh' % (sbatch_files[i], ii) for ii in range(n_splits)]
        
        job_time = get_job_attribute([sbatch_files[i]], '-t', exp_type = exp_type)[0]
        jobname = get_job_attribute([sbatch_files[i]], '--job-name', exp_type = exp_type)[0]
        script_dir = '/global/homes/a/akumar25/repos/uoicorr'
        sbatch_array = []
        for j, sbatch_file in enumerate(split_sbatch_files):
            s = {
            'arg_file' : split_param_files[j],
            'ntasks' : 34,            
            'exp_type' : exp_type,
            'job_time' : job_time,
            'sbname' : 'sbatch%d_split%d.sh' % (jobnums[i], j), 
            'jobname' : '%s_split%d' % (jobname, j)
            }
            sbatch_array.append(s)

        generate_sbatch_scripts(sbatch_array, '%s/%s' % (jobdir, exp_type), script_dir)
        
# Back out a submit file from the log file. 
# logfile: path to log
def submit_from_log(logfile, submit_file):

    # Load the log file
    log = pd.read_pickle(logfile)

    # For each column, gather the unique values
    unique_attributes = {}
    for col in list(logfile.columns):
        unique_attributes[col] = unique_obj(np.array(log[col].values).ravel())

    # The sub_iter_params provide some clue as to which things one iterated
    # over within a given arg_file. The assumption we make is that any unique
    # values with > 1 element (except l1_ratios)

    iter_params = {}

    excluded_keys = ['l1_ratios']

    if 'sub_iter_params' in list(unique_attributes.keys()):
        
        excluded_keys.extend(unique_attributes['sub_iter_params'])

    for key, value in unique_attributes.items():

        if key not in excluded_keys and not np.isscalar(value):

            iter_params[key] = value

    # Remove the keys in iter_params from unique_attributes and let these
    # be the comm_params

    comm_params = unique_attributes.copy()
    for key in list(iter_params.keys()): 
        del comm_params[key]

    pdb.set_trace()

    # Write to file
    with open('%s.py' % submit_file, 'w') as f:       

        # Some default lines which can be changed after the fact
        f.write('import numpy as np\n')
        f.write('import itertools\n')
        f.write('import pdb\n')
        f.write('from misc import get_cov_list\n')
        f.write('\n')

        f.write("script_dir  = '/global/homes/a/akumar25/repos/uoicorr'\n")
        f.write("exp_types = ['UoILasso']\n")
        f.write("algorithm_times = ['24:00:00']\n")

        # repr produces the string corresponding to the variable stdout. Then, 
        # use reg expr to insert a newline character prior to every new dictionary
        # key for nice formatting   
        iter_param_string = repr(iter_params)

        for k in list(iter_params.keys()):
            iter_param_string.insert('\n', iter_param_string.find(k))

        f.write('iter_params = %s\n' % iter_param_string)

        # same thing for comm_params
        comm_param_string = repr(comm_params)

        for k in list(comm_params.keys()):
            comm_param_string.insert('\n', comm_param_string.find(k))

        f.write('comm_params = %s\n' % comm_param_string)



