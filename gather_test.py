import sys, os
from mpi4py import MPI

# Crawl up to the repos folder
while current_dir not in ['repos', 'nse']:
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
data = np.random.random_integers(size = (1000))

# Gather data
data = Gatherv_rows(send = data, comm, root = 0)

print('Data gathered!')