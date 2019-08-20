import sys, os
import pdb
import itertools
import glob
import argparse
import pickle
import struct
import numpy as np
from mpi4py import MPI
import h5py_wrapper
import time
from pydoc import locate
from sklearn.preprocessing import StandardScaler

from mpi_utils.ndarray import Bcast_from_root, Gatherv_rows, Gather_ndlist

from job_utils.results import  ResultsManager
from job_utils.idxpckl import Indexed_Pickle

from utils import gen_covariance, sparsify_beta, gen_data
from results_manager import init_results_container, calc_result

def manage_comm():

    '''Create a comm object and split it into the appropriate number of subcomms'''

    comm = MPI.COMM_WORLD
    rank = comm.rank
    numproc = comm.Get_size()

    if args.comm_splits is None:
        if args.exp_type in ['UoILasso', 'UoIElasticNet']:
            comm_splits = 1
        else:
            comm_splits = numproc
    else:
        comm_splits = args.comm_splits
    # Use array split to do comm.split

    ranks = np.arange(numproc)
    split_ranks = np.array_split(ranks, comm_splits)
    color = [i for i in np.arange(comm_splits) if rank in split_ranks[i]][0]
    subcomm_roots = [split_ranks[i][0] for i in np.arange(comm_splits)]

    subcomm = comm.Split(color, rank)

    nchunks = comm_splits
    subrank = subcomm.rank
    numproc = subcomm.Get_size()

    # Create a group including the root of each subcomm (unused at the moment)
    # global_group = comm.Get_group()
    # root_group = MPI.Group.Incl(global_group, subcomm_roots)
    # roots_comm = comm.Create(root_group)

    return comm, rank, color, subcomm, subrank, numproc, comm_splits


def gen_data_(params, subcomm, subrank):
    ''' Use the seeds provided from the arg file to generate regression design and data'''

    seed = params['seed']

    if subrank == 0:
        # Generate covariance
        sigma = gen_covariance(params['n_features'],
                               params['cov_params']['correlation'], 
                               params['cov_params']['block_size'],
                               params['cov_params']['L'],
                               params['cov_params']['t'])


        # Sparsify the beta - seed with the block size
        beta = sparsify_beta(params['betadict']['beta'], params['cov_params']['block_size'],
                             params['sparsity'], seed=params['cov_params']['block_size'])

    else:
        
        sigma = None
        beta = None

    sigma = Bcast_from_root(sigma, subcomm)
    beta = Bcast_from_root(beta, subcomm)

    params['sigma'] = sigma
    params['betas'] = beta      

    # If all betas end up zero for this sparsity level, output a warning and skip
    # this iteration (Make sure all ranks are instructed to continue!)
    if np.count_nonzero(beta) == 0:
        print('Warning, all betas were 0!')
        print(params)

    if subrank == 0:

        # Generate data
        X, X_test, y, y_test, ss = gen_data(params['n_samples'], params['n_features'],
                                        params['kappa'], sigma, beta, seed)

        # Standardize
        X = StandardScaler().fit_transform(X)
        X_test = StandardScaler().fit_transform(X_test)
        y -= np.mean(y)
        y_test -= np.mean(y_test)

    else:
        X = None
        X_test = None
        y = None
        y_test = None
        ss = None


    X = Bcast_from_root(X, subcomm)
    X_test = Bcast_from_root(X_test, subcomm)
    y = Bcast_from_root(y, subcomm)
    y_test = Bcast_from_root(y_test, subcomm)
    ss = Bcast_from_root(ss, subcomm)

    params['ss'] = ss

    return X, X_test, y, y_test, params

def main(args):
    total_start = time.time()

    # Open the arg file and read out metadata
    f = Indexed_Pickle(args.arg_file)
    f.init_read()
    total_tasks = len(f.index)

    n_features = f.header['n_features']

    exp_type = args.exp_type
    results_dir = args.results_dir

    # MPI initialization 
    comm, rank, color, subcomm, subrank, numproc, comm_splits = manage_comm()

    # Load or initialize the Results Manager object
    if args.resume:
        rmanager = ResultsManager.restore_from_directory(results_dir)
    else:
        rmanager = ResultsManager(total_tasks = total_tasks, directory = results_dir)

    # Only have one process handle file I/O
    if rank == 0:
        rmanager.makedir()

    # Chunk up iter_param_list to distribute across iterations. 

    # Take the complement of inserted_idxs in the results manager
    task_list = np.array(list(set(np.arange(total_tasks)).difference(set(rmanager.inserted_idxs()))))
    chunk_param_list = np.array_split(np.arange(len(task_list)), comm_splits)
    chunk_idx = color
    num_tasks = len(chunk_param_list[chunk_idx])

    print('total tasks %d' % len(task_list))
    print('rank: %d, subrank: %d' % (comm.rank, subrank))
    print('num_tasks: %d' % num_tasks)

    # Initialize arrays to store data in. Assumes that n_features
    # is held constant across all iterations

    # hard-code n_reg_params because why not
    if exp_type in ['EN', 'scad', 'mcp']: 
        n_reg_params = 2
    else: 
        n_reg_params = 1

    for i in range(num_tasks):
        start = time.time()

        params = f.read(chunk_param_list[chunk_idx][i])

        params['comm'] = subcomm

        X, X_test, y, y_test, params = gen_data_(params, subcomm, subrank)

        # Hard-coded convenience for SCAD/MCP
        if exp_type in ['scad', 'mcp']:
            exp = locate('exp_types.%s' % 'PYC')
            params['penalty'] = exp_type
        else: 
            exp = locate('exp_types.%s' % exp_type)

        exp_results = exp.run(X, y, params)
        if subrank == 0:

            # Directly save exp_results

            # Add results to results manager. This automatically saves the child's data
            rmanager.add_child(exp_results, idx = chunk_param_list[chunk_idx][i])
            print('Process group %d completed outer loop %d/%d' % (rank, i, num_tasks))
            print(time.time() - start)

        del params

        if args.test and i == args.ntest:
            break

    f.close_read()

    # gather results managers
    rmanager.gather_managers(comm)

    if rank == 0:
        # concatenate and clean up results
        # rmanager.concatenate()
        # rmanager.cleanup()
        pass
    print('Total time: %f' % (time.time() - total_start))
    print('Job completed!')

if __name__ == '__main__': 

    total_start = time.time()

    ###### Command line arguments #######
    parser = argparse.ArgumentParser()

    parser.add_argument('arg_file')
    parser.add_argument('results_dir', default = './mpi_submit_test')
    parser.add_argument('exp_type', default = 'UoILasso')
    parser.add_argument('--comm_splits', type=int, default = None)
    parser.add_argument('-t', '--test', action = 'store_true')
    # Number of reps to break after if we are just testing
    parser.add_argument('--ntest', type = int, default = 1)

    # Does this job need to be resumed?
    parser.add_argument('-r', '--resume', action='store_true')

    args = parser.parse_args()

    #######################################
    main(args)
