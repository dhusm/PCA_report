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

from optparse import OptionParser
import csv, sys, copy, os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.optimize as opt
from matplotlib import rc
import utility
style.use('bmh')
cwd = os.getcwd()

def mu_local(xyz,wxyz,mu0):
    mu_xyz = mu0 - 0.5*m6Li*(np.dot(xyz**2,wxyz**2))
    return mu_xyz
    
px = 6.45 / 1.56 * 1e-6
tof = 1e-3

nuFortPower = 0.8 ##Druing imaging
nuFort=dipolepotentials.Dipoletrap(waistx=64.7e-6,waisty=80.e-6,wavelength=1064e-9,power=nuFortPower)	#waists chosen such that for a power of 0.150W, nux=194Hz and nuz=157Hz, as measured for the quantized conductance paper
trapFreqs=nuFort.trap_frequencies(dipolepotentials.Li6)
wx = trapFreqs[0]*2*pi
wz = trapFreqs[1]*2*pi
wy = 31.1*2*pi
wxyz = np.array([wx,wy,wz])
kT = 100e-9*kb
U0 = -nuFort.potential(dipolepotentials.Li6)
ls_pos = 80

xyz = np.array([0,0,0])
# Ballistic into saddlepoint
bx,by,bz = sqrt(1+(wxyz*tof)**2)
by=1
# Hydrodynamic into saddlepoint
## bx = 1.22*wx*tof
## by = 2.05*pi/2.*tof*sqrt(wz*wx)*(wy/sqrt(wz*wx))**2
## bz = 1.22*wz*tof
#by=1

height = 150
width = 50

x_sample = np.arange(-width/2*px,width/2*px,px) / bx
y_sample = np.arange(-height/2*px,height/2*px,px) / by
z_sample = np.arange(-width/2*px,width/2*px,px) / bz
X,Y = np.meshgrid(x_sample,y_sample)
n = np.zeros([len(x_sample),len(y_sample),len(z_sample)])
v = np.zeros([len(x_sample),len(y_sample),len(z_sample)])
#q = mu_loc/kT
mu0 = 0.25e-6*kb

fname = 'test_cloud.npy'

## cloud_sampled = os.path.exists('\\'.join([cwd, fname]))
## force_cloud = True
## if (cloud_sampled and (not force_cloud)):
##     with open(fname,'rb') as f:
##         n = np.load(fname)
##         print 'Found sampled cloud file. Loading..'
## else:
##     print 'File not found, re-sampling full cloud...'
##     n=np.array([[[utility.get_loc_unit_density(kT,mu_local(np.array([x,y,z]),wxyz,mu0) /kT) /(bx*by*bz)
##             for x in x_sample]
##             for y in y_sample] 
##             for z in z_sample])
##     print 'Done!'
##     with open(fname,'wb') as f:
##         np.save(f,n)
        



n *= px**3
#%%
n2d = np.sum(n,axis=0)
n1d = np.sum(n2d,axis=1)

## 1/0
## reps = 100
## noise = np.array([np.random.normal(0,1,shape(n2d)) for i in range(reps)])
## n2d_noise = np.zeros(shape(noise))
## noise_amp = 0.1*(np.amax(n2d) - np.amin(n2d))
## noise *= noise_amp
## n2d_noise = n2d + noise
## n1d_noise = np.sum(n2d_noise,axis=2)
close('all')
fig,ax = subplots(1,2)
im=ax[0].imshow(n2d,cmap='Greys_r')
## im2=ax[1].imshow(n2d_noise[0],cmap='Greys_r')
## im2.set_clim(im.get_clim())
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
fig.colorbar(im,cax=cbar_ax)
fig.savefig('Cloud_mu_' + str(int(mu0/kb*1e9))+'_T_' + str(int(kT/kb*1e9)) + '.pdf')

##Start Second moment analysis
ROI_pca_sm = [0,height,0,width]
ROI_pca_sm_offset = [0,height,0,width]
sm = analysis.calc_secondMoment_1d(n2d, ROI_pca_sm_offset,ROI_pca_sm)

Ntot = np.sum(n2d)
T_rec = utility.get_T_Unitary(wx,wy,wz,U0,sm,Ntot)
1/0



#%%
my_PCA = PCA_decomposition(n2d_noise)
my_PCA.get_eigensystem(doReduced=True)
rec_pic = [my_PCA.reconstruct_from_PCA(pic,3) for pic in n2d_noise]
#%%

sm = analysis.calc_secondMoment_1d(n2d, ROI_pca_sm_offset,ROI_pca_sm)
sm_noise = np.zeros(len(n2d_noise))
sm_rec = np.zeros(len(n2d_noise))
for i,npic in enumerate(n2d_noise):
    sm_noise[i] = analysis.calc_secondMoment_1d(npic, ROI_pca_sm_offset,ROI_pca_sm)
    sm_rec[i] = analysis.calc_secondMoment_1d(rec_pic[i], ROI_pca_sm_offset,ROI_pca_sm)
#%%
sm_up,sm_down = analysis.calc_secondMomentUp_secondMomentDown_1d(n2d,ls_pos, ROI_pca_sm_offset,ROI_pca_sm,forcePlot=True,crop_barrier=2)
#%%
Ntot = np.sum(rec_pic[0])
T_rec = np.array([utility.get_T_Unitary(wx,wy,wz,U0,my_sm,Ntot) for my_sm in sm_rec])
T_noise = np.array([utility.get_T_Unitary(wx,wy,wz,U0,my_sm,Ntot) for my_sm in sm_noise])
T_orig = utility.get_T_Unitary(wx,wy,wz,U0,sm,Ntot)
#%%
fig,ax = subplots()
ax.plot(sm_noise,'b')
ax.plot(sm_rec,'r')
ax.axhline(sm)
show()   
#%%
fig_T,ax_T = subplots()
ax_T.plot(T_rec*1e9,'r-',label='reconstructed')
ax_T.plot(T_noise*1e9,'b-',label='noisy')
ax_T.axhline(kT/kb*1e9,color='k')
ax_T.axhline(T_orig*1e9,color='grey')
ax_T.set_ylabel('Temperature (nK)')
fig_T.savefig('TemperaturesSM_mu_' + str(int(mu0/kb*1e9))+'_T_' + str(int(kT/kb*1e9)) + '.pdf')
