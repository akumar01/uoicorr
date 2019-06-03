import os, glob, sys
import json
import h5py
import pickle
import numpy as np
import pandas as pd
import itertools
import importlib
import struct
import pdb
import time
from job_manager import grab_files


# Common postprocessing operations on a single data file
def postprocess(data_file, param_file, fields = None):

    data_list = []

    # Load pickled data file. It should contain a single list with the
    # same length as index
    with open(data_file, 'rb') as f:
        sa = pickle.load(f)

    # Indexed pickle file
    param_file.seek(0, 0)
    index_loc = param_file.read(8)
    index_loc = struct.unpack('L', index_loc)[0]
    total_tasks = pickle.load(param_file)
    n_features = pickle.load(param_file)
    param_file.seek(index_loc, 0)
    index = pickle.load(param_file)
    for i, loc in enumerate(index):
        
        param_file.seek(loc, 0)
        params = pickle.load(param_file)
        data_dict = params.copy()
        # Do not store Sigma or beta to save memory
        data_dict['sigma'] = []
        data_dict['beta'] = []
        data_dict['sa'] = sa[i]

        data_list.append(data_dict)
    return data_list

# Postprocess specifically output of mpi_submit_estimate_sa, where the 
# data is stored as a pickle file. Otherwise, works largely like the normal
# postprocess.py
def postprocess_dir(jobdir, exp_type = None):
    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.dat', exp_type)
    # List to store all data
    data_list = []
    for data_file in data_files:
        _, fname = os.path.split(data_file)
        jobno = fname.split('.dat')[0].split('job')[1]
        with open('%s/master/params%s.dat' % (jobdir, jobno), 'rb') as f:
                d = postprocess(data_file, f)
                data_list.extend(d)
    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)

    print(dataframe.shape)
    return dataframe
