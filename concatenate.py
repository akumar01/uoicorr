from glob import glob 
import argparse
import h5py_wrapper


# Sequentially load all results and concatenate into a single file
def main(path):

    # Expected number of tasks:
    expected_total_tasks = int(glob('%s/totaltasks*')[0].split('_')[-1])

    # Grab all files with purely numeric filenames
    datfiles = glob('%s/*.dat' % path)
    filenames = [df.rpartition('/')[-1].split('.dat')[0] for df in datfiles]
    numeric_datfiles = [df for i, df in enumerate(datfiles) if filenames[i].isdigit()]

    if len(numeric_datfiles) == 0:
        print('Warning! No sub-task data files found in specified directory.')

    # First check if we have already tried to concatenate results together
    concat_file = glob('%s/concat.dat' % path) 
    if len(concat_file) == 0:
        # No concatenation has been done yet

        # Load the first datfile to obtain the dictionary structure
        dummy_df = h5py_wrapper.load(numeric_datfiles[0])
        concat = init_master_results(dummy_df)
    else:
        concat = h5py_wrapper.load(concat_file[0])

    # Next, sequentially open up numeric_datfiles and insert them into the concatenated results file   
    for datfile in numeric_datfiles:
        concat = insert_results


if __name__ == '__main__':
    # Read the directory to operate on from cmd line args 
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.parse_args()
    main(parser.path)
