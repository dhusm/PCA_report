##########################################
# Template for analysing transport data from X-Cam
# Only processing, no loading and reading. But you can use the data of several runs at once
#
# Version from 2014 - 09 - 16
# created by S. Krinner
##########################################

import numpy, pickle

from pylab import *
from time import time
from scipy.optimize import leastsq,curve_fit
from qo.expwiz.fits import FitFermi, FitCount
from qo.evaltools import analysis
from qo.theory import dipolepotentials
from qo.constants import *
import pandas as pd
import sys
from PCA import PCA_decomposition
from qo.expwiz.imagefile import ImageFile

from optparse import OptionParser
import csv, sys, copy, os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.optimize as opt
from matplotlib import rc
import utility

def mu_local(xyz,wxyz,mu0):
    mu_xyz = mu0 - 0.5*m6Li*(np.dot(xyz**2,wxyz**2))
    return mu_xyz
    
px = 6.45 / 1.56 * 1e-6
tof = 1e-3

nuFortPower = 0.8
nuFort=dipolepotentials.Dipoletrap(waistx=64.7e-6,waisty=80.e-6,wavelength=1064e-9,power=nuFortPower)	#waists chosen such that for a power of 0.150W, nux=194Hz and nuz=157Hz, as measured for the quantized conductance paper
trapFreqs=nuFort.trap_frequencies(dipolepotentials.Li6)
wx = trapFreqs[0]*2*pi
wz = trapFreqs[1]*2*pi
wy = 31.1*2*pi
wxyz = np.array([wx,wy,wz])
kT = 200e-9*kb
U0 = -nuFort.potential(dipolepotentials.Li6)
ls_pos = 120

pic = ImageFile.from_file("XGlassDensity_Imaging_Detection_Det_-24.png").get_data()

#%%
ROI_pca_sm = [0,240,0,80]
ROI_pca_sm_offset = [0,240,0,80]

sm = analysis.calc_secondMoment_1d(pic, ROI_pca_sm_offset,ROI_pca_sm)

#%%
sm_up,sm_down = analysis.calc_secondMomentUp_secondMomentDown_1d(pic,ls_pos, ROI_pca_sm_offset,ROI_pca_sm,forcePlot=True,crop_barrier=2)
#%%
Ntot = np.sum(pic)
T_orig = utility.get_T_Unitary(wx,wy,wz,U0,sm,Ntot)

    
    