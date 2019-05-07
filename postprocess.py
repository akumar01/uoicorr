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
from job_manager import grab_files

# Common postprocessing operations on a single data file
def postprocess(data_file, param_file, fields = None):

    data_list = []

    # Indexed pickle file
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
        # Do not store Sigma to save memory
        data_dict['sigma'] = []

        if fields is None:
            for key in data_file.keys():
               data_dict[key] = data_file[key][:] 
        else:
            for key in fields:
                data_dict[key] = data_file[key][:]

        data_list.append(data_dict)
    return data_list

# Postprocess an entire directory of data, will assume standard nomenclature of
# associated parameter files
# old_format: back when we were using the .py param files
# skip_bad: Skip over files that cannot be processed without raising errors
# arg_flag: Only return dataframes for data files that match arg_flag {key: value}
def postprocess_dir(jobdir, exp_type = None, fields = None):
    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.h5', exp_type)
    # List to store all data
    data_list = []

    for data_file in data_files:

        _, fname = os.path.split(data_file)
        jobno = fname.split('.dat')[0].split('job')[1]
        with h5py.File(data_file, 'r') as f1:
            with open('%s/master/params%s.dat' % (jobdir, jobno)) as f2:
                d = postprocess(f1, f2, fields)
                data_list.extend(d)

    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)

    print(dataframe.shape)
    return dataframe
