import sys, os
from mpi4py import MPI
import numpy as np
parent_path, current_dir = os.path.split(os.path.abspath('.'))

# Crawl up to the repos folder
while current_dir not in ['repos', 'nse']:
    print('In loop!')
    parent_path, current_dir = os.path.split(parent_path)



p = os.path.join(parent_path, current_dir)

# Add uoicorr and pyuoi to the path
if '%s/uoicor' % p not in sys.path:
    sys.path.append('%s/uoicorr' % p)
if '%s/PyUoI' % p not in sys.path:
    sys.path.append('%s/PyUoI' % p)

from pyuoi.mpi_utils import Gatherv_rows

# Initialize comm object
comm = MPI.COMM_WORLD
rank = comm.rank
numproc = comm.Get_size()

# Generate some random data
data = np.random.random_integers(0, 100, size = (1000))

print('Starting gather')

# Gather data
data = Gatherv_rows(data, comm, root = 0)

print('Data gathered!')
