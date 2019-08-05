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

# Eventually turn this into its own standalone storage solution
class Indexed_Pickle():
    
    def __init__(self, file):
        self.file = file
        file.seek(0, 0)
        index_loc = file.read(8)
        # Some weird Windows bullshit
        if os.name == 'nt':
            index_loc = struct.unpack('q', index_loc)[0]
        else:
            index_loc = struct.unpack('L', index_loc)[0]
        total_tasks = pickle.load(file)
        n_features = pickle.load(file)
        file.seek(index_loc, 0)
        self.index = pickle.load(file)
        self.index_length = total_tasks
   
    def read(self, idx):
        
        self.file.seek(self.index[idx], 0)
        data = pickle.load(self.file)
        return data

# Common postprocessing operations on a single data file
def postprocess(data_file, param_file, fields = None):
    data_list = []

    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)
    
    for i in range(param_file.index_length):
        params = param_file.read(i)
        data_dict = params.copy()
        # Do not store Sigma to save memory
        data_dict['sigma'] = []
        if fields is None:
            for key in data_file.keys():
                data_dict[key] = data_file[key][i] 
        else:
            for key in fields:
                data_dict[key] = data_file[key][i]

        data_list.append(data_dict)

    return data_list

# New format with results from multiple selection methods
def postprocess_v2(data_file, param_file, fields = None):
    
    data_list = []
    
    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)
    
    for i in range(param_file.index_length):
        params = param_file.read(i)
        # Enumerate over selection methods and save a separate pandas row for each selection method
        selection_methods = list(data_file.keys())
        for selection_method in selection_methods:
            data_dict = params.copy()
            data_dict['selection_method'] = selection_method 
            # Save memory
            data_dict['sigma'] = []

            if fields is None:
                for key in data_file[selection_method].keys():
                    data_dict[key] = data_file[selection_method][key][i]
            else:
                for key in fields:
                    data_dict[key] = data_file[selection_method][key][i]
            data_list.append(data_dict)
    return data_list

# Postprocess an entire directory of data, will assume standard nomenclature of
# associated parameter files
# exp_type: only postprocess results for the given exp_type
# fields (list): only return data for the fields given in fields (useful for saving
# memory)
# old format: Use postprocess instead of postprocess_v2
def postprocess_dir(jobdir, exp_type = None, fields = None, old_format = False):
    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.dat', exp_type)
    print(len(data_files))
    # List to store all data
    data_list = []
    for i, data_file in enumerate(data_files):
        _, fname = os.path.split(data_file)
        jobno = fname.split('.dat')[0].split('job')[1]
        with h5py.File(data_file, 'r') as f1:
            with open('%s/master/params%s.dat' % (jobdir, jobno), 'rb') as f2:
                pdb.set_trace()
                d = postprocess_v2(f1, f2, fields)
                data_list.extend(d)        
        print(i)
        
    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)

    print(dataframe.shape)
    return dataframe
