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

def main(args): 
    total_start = time.time()

    exp_type = args.exp_type
    results_file = args.results_file

    # Create an MPI comm object
    comm = MPI.COMM_WORLD
    rank = comm.rank
    numproc = comm.Get_size()

    # If specified, split the comm object into subcommunicators. The number of splits will
    # determine the number of parallel executions of the outer loop. This is useful for UoI
    # to parallelize over both bootstraps and repetitions

    if args.comm_splits is None:
        if exp_type in ['UoILasso', 'UoIElasticNet']:
            args.comm_splits = 1
        else:
            args.comm_splits = numproc

    # Use array split to do comm.split
    ranks = np.arange(numproc)
    split_ranks = np.array_split(ranks, args.comm_splits)
    color = [i for i in np.arange(args.comm_splits) if rank in split_ranks[i]][0]
    subcomm_roots = [split_ranks[i][0] for i in np.arange(args.comm_splits)]

    subcomm = comm.Split(color, rank)

    rank = color
    nchunks = args.comm_splits
    subrank = subcomm.rank
    numproc = subcomm.Get_size()

    # Create a group including the root of each subcomm.
    global_group = comm.Get_group()
    root_group = MPI.Group.Incl(global_group, subcomm_roots)
    roots_comm = comm.Create(root_group)

    # Open the arg file and read out the index array, number of
    # total_tasks and n_features

    f = open(args.arg_file, 'rb')
    index_loc = f.read(8)
    index_loc = struct.unpack('L', index_loc)[0]
    total_tasks = pickle.load(f)
    n_features = pickle.load(f)
    f.seek(index_loc, 0)
    index = pickle.load(f)

    # Chunk up iter_param_list to distribute across iterations
    chunk_param_list = np.array_split(np.arange(total_tasks), nchunks)
    chunk_idx = rank
    num_tasks = len(chunk_param_list[chunk_idx])

    print('rank: %d, subrank: %d, color: %d' % (comm.rank, subrank, color))
    print('num_tasks: %d' % num_tasks)

    # Initialize arrays to store data in. Assumes that n_features
    # is held constant across all iterations

    #selection_methods = ['CV', 'AIC', 'BIC', 'eBIC', 'OIC']
    selection_methods = ['BIC']

    # hard-code n_reg_params because why not
    if exp_type in ['EN', 'scad', 'mcp']: 
        n_reg_params = 2
    else: 
        n_reg_params = 1

    if subrank == 0:

        fields = ['FNR', 'FPR', 'sa', 'ee', 'median_ee', 'r2', 'beta_hats', 
                'MSE', 'reg_param', 'oracle_penalty']

        results_dict = init_results_container(selection_methods, fields, 
                                              num_tasks, n_features,
                                              n_reg_params)

    for i in range(num_tasks):
        start = time.time()

        f.seek(index[chunk_param_list[chunk_idx][i]], 0)
        params = pickle.load(f)

        # Skip this guy because for the particular combination of parameters,
        # all betas end up being zero
        if 'skip' in list(params.keys()):
            if params['skip']:
                continue

        params['comm'] = subcomm
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
            continue

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

        # Hard-coded convenience for SCAD/MCP
        if exp_type in ['scad', 'mcp']:
            exp = locate('exp_types.%s' % 'PYC')
            params['penalty'] = exp_type
        else: 
            exp = locate('exp_types.%s' % exp_type)

        exp_results = exp.run(X, y, params, selection_methods)

        if subrank == 0:
            #### Calculate and log results for each selection method
            for selection_method in selection_methods:

                for field in fields:
                    results_dict[selection_method][field][i] = calc_result(X, X_test, y, y_test,
                                                                           beta.ravel(), field,
                                                                           exp_results[selection_method])

            print('Process group %d completed outer loop %d/%d' % (rank, i, num_tasks))
            print(time.time() - start)

        del params
        if args.test and i == args.ntest:
            break

    # Gather across root nodes (using Gatherv rows)
    if subrank == 0:

        results_dict = gather_results(results_dict, roots_comm)


    f.close()
    if comm.rank == 0:

        h5py_wrapper.save(results_file, results_dict, write_mode = 'w')

        print('Total time: %f' % (time.time() - total_start))
        print('Job completed!')

if __name__ == '__main__':
    ###### Command line arguments #######
    parser = argparse.ArgumentParser()

    parser.add_argument('arg_file')
    parser.add_argument('results_file', default = 'results.h5')
    parser.add_argument('exp_type', default = 'UoILasso')
    parser.add_argument('--paths', nargs='+', type=str)
    parser.add_argument('--comm_splits', type=int, default = None)
    parser.add_argument('-t', '--test', action = 'store_true')
    # Number of reps to break after if we are just testing
    parser.add_argument('--ntest', type = int, default = 1)
    args = parser.parse_args()
    #######################################

    # Append paths before importing uoicorr scripts
    for path in args.paths:
        sys.path.append(path)

    from utils import gen_covariance, gen_data, sparsify_beta
    from results_manager import init_results_container, calc_result, gather_results

    main(args)



