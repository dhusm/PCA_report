import numpy, pickle

from pylab import *
from qo.constants import *
import pandas as pd
from PCA import PCA_decomposition

import csv, sys, copy, os
import scipy.optimize as opt
import utility
from scipy.stats import moment

## style.use('bmh')
cwd = os.getcwd()

home = os.path.expanduser("~")
style.use('C:\\Users\\d_hus\\.matplotlib\\stylelib\\fountain.mplstyle')

df = pd.read_pickle(cwd + '\\nOD_N_kT_sampled')
x = np.unique(df.Ntot.values)
y = np.unique(df.Temp.values)
df = df.set_index(['Ntot','Temp'])
df = df.unstack('Temp')

close('all')
fig,ax = subplots(figsize=(2.5,2))
Y, X = np.meshgrid(y,x*1e-3)

im = ax.pcolormesh(X,Y,df.sm.values)
ax.contour(X,Y,df.sm.values,colors='k',levels=[200,250,300,350,400,450,500,550])
colorbar(im,label='sm')
ax.set_xlabel(r'$N\ (10^3$ atoms)')
ax.set_ylabel(r'$T$ (nK)')
fig.tight_layout()
fig.savefig('sm_kT_N_sampled.pdf')
show()