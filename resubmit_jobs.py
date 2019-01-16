import sys, os
import argparse

# Resubmit all the jobs in a given directory
parser = argparse.ArgumentParser()
parser.add_argument('directory', default = None)

a = parser.parse_args()

