import numpy, pickle

from pylab import *
import pandas as pd
from PCA import PCA_decomposition

import csv, sys, copy, os
from matplotlib import rc
import utility
from scipy.stats import moment

style.use('bmh')
cwd = os.getcwd()

home = os.path.expanduser("~")
style.use(home+'\\.matplotlib\\stylelib\\fountain.mplstyle')

df = pd.read_pickle('mu_300_T_495.pkl')

my_pca = PCA_decomposition(df.nOD)
my_pca.get_eigensystem()


close('all')
f1,a1 = subplots(1)
a1.plot(my_pca.val)
a1.set_yscale('log')
show()