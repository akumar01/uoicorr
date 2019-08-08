#Script to import an R package
from rpy2.robjects.packages import importr

pckg = importr('PACKAGE')