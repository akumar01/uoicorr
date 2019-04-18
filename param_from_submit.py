import numpy as np
import itertools
import sys
import importlib
import pickle

# Generate a single short arg file from a submit_file
def single_from_submit(submit_file, name,
                        exp_type = None, generate_sbatch = False):
    path, name = arg_file.split('/')
    sys.path.append(path)
    args = importlib.import_module(name)

    iter_params = args.iter_params
    comm_params = args.comm_params
    exp_types = args.exp_types
    algorithm_times = args.algorithm_times
    script_dir = args.script_dir

    # From the combination of exp_types and iter_params, select
    # only a single combination
    if exp_type is None:
        exp_type = exp_types[0]

    algorithm_time = algorithm_times[exp_types.index[]]

    iter_keys = list(iter_params.keys())

    task_array = []
    for j, arg_comb in enumerate(itertools.product(*list(iter_params.values()))):
        arg = {}
        for j in range(len(arg_comb)):
            arg[iter_keys[j]] = arg_comb[j]
        for key, value in comm_params.items():
            arg[key] = value
        arg['exp_type'] = exp_type

        task_array.append(arg)

    # Pick one at random
    random_arg = np.random.choice(task_array)

    result_file = name + '.h5'
    random_arg['result_file'] = result_file
    # Write it to file:
    arg_file = name + '_params.dat'

    with open(arg_file, 'wb') as f:
        f.write(pickle.dumps)


    # Also generate a template sbatch file for batch submmission
    if generate_sbatch:
        sbname = name + '.sh'
        script = 'mpi_submit.py'

        if exp_type in ['UoIElasticNet', 'UoILasso']:
            nprocs = 48
        elif exp_type in ['GTV']:
            nprocs = 272
        else:
            # Number of sub_iter tasks
            nprocs = \
            len(args['reps'] * list(itertools.product(*[args[key] for key in args['sub_iter_params']])))
        with open(sbname, 'w') as sb:
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
            sb.write('export OMP_PLACES=threads\n')
            sb.write('export OMP_PROC_BIND=spread\n')

            sb.write('srun -n %d -c 1 python3 -u %s %s' 
                    % (nprocs, script, arg_file))



        #!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 272 -c 1 --cpu_bind=threads myapp.x