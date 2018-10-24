#########################################
##      Create 1D density profiles for sampeled atom number an temperature
#########################################
# DH / 22.10.2018
 
import numpy, pickle

from pylab import *
from qo.theory import dipolepotentials
from qo.constants import *
from qo.evaltools import analysis
import pandas as pd
from PCA import PCA_decomposition

from optparse import OptionParser
import csv, sys, copy, os
import scipy.optimize as opt
import utility
from scipy.stats import moment

## style.use('bmh')
cwd = os.getcwd()

home = os.path.expanduser("~")
style.use(home + '\\.matplotlib\\stylelib\\fountain.mplstyle')


px = 6.45e-6 / 1.56
height = 200
y_sample = np.arange(-height/2*px,height/2*px,px)

def n1d(y,wr,wy,lT,kT,q0):
    n = (2.*pi/(m6Li*wr**2))*(kT/lT**3) * utility.get_fp(q0-0.5/kT*m6Li*(wy*y)**2)
    return n
    
def secMom(y,n):
    sm = np.sum(y**2*n)/np.sum(n)
    return sm

def harmonic_trap_nOD(y,kT,mu,wr,wy,noise=0):
    lT = np.sqrt(h**2/(2*pi*m6Li*kT))
    n = n1d(y_sample,wr,wy,lT,kT,mu/kT) ## atoms line density
    noise_eff = np.amax(n)*noise
    y_noise = (np.random.rand(len(y))-0.5)*noise_eff
    n += y_noise
    Ntot = np.sum(n)*px
    nOD = n* sigma0 ## from atom density to optical density
    sm = secMom(y_sample/px,nOD) ## in px**2
    T_sm,ttf = utility.get_T_Unitary(wx,wy,wz,U0,sm,Ntot)
    return nOD, T_sm, sm, Ntot,ttf
    
def get_N(mu,y,kT,wr,wy):
    lT = np.sqrt(h**2/(2*pi*m6Li*kT))
    n = n1d(y_sample,wr,wy,lT,kT,mu/kT) ## atoms line density
    Ntot = np.sum(n)*px
    return Ntot
    
def find_mu(y,kT,Ntot,wr,wy):
    f_opt = lambda mu: np.abs(get_N(mu*1e-9*kb,y_sample,kT,wr,wy) - Ntot)
    popt = opt.minimize_scalar(f_opt)
    mu = popt.x*1e-9*kb
    return mu
    
    
nuFortPower = 0.8 ##Druing imaging
nuFort=dipolepotentials.Dipoletrap(waistx=64.7e-6,waisty=80.e-6,wavelength=1064e-9,power=nuFortPower)	
trapFreqs=nuFort.trap_frequencies(dipolepotentials.Li6)
wx = trapFreqs[0]*2*pi
wz = trapFreqs[1]*2*pi
wr = np.sqrt(wx*wz)
wy = 31.1*2*pi
wxyz = np.array([wx,wy,wz])
U0 = -nuFort.potential(dipolepotentials.Li6)
sigma0 = 0.5*3*0.67e-6**2/2/pi

SNR = 0.

df = pd.DataFrame(columns=['Temp','sm','mu','nOD','Ntot','EE0','EF','ttf'])

kT_list = np.arange(50,300,5)*1e-9*kb
N_list = np.arange(30e3,200e3,2.5e3)
i = 0
for my_Ntot in N_list:
    print my_Ntot
    for kT in kT_list:        
        mu = find_mu(y_sample,kT,my_Ntot,wr,wy)
        nOD, T_sm, sm, Ntot,ttf = harmonic_trap_nOD(y_sample,kT,mu,wr,wy,noise=SNR)
        EE0,Ef = analysis.calc_EoverE0(wx,wy,wz,U0,sm,Ntot)
        
        df.loc[i] = [kT/kb*1e9,sm,mu,np.array(nOD,dtype='float'),np.round(ceil(Ntot),-2),EE0,Ef,ttf]
        i += 1

df.to_pickle(cwd + '\\nOD_N_kT_sampled')