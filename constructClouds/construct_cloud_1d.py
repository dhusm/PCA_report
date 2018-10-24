import numpy, pickle

from pylab import *
from qo.theory import dipolepotentials
from qo.constants import *
from qo.evaltools import analysis
import pandas as pd
from PCA import PCA_decomposition

from optparse import OptionParser
import csv, sys, copy, os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.optimize as opt
from matplotlib import rc
import utility
from scipy.stats import moment

style.use('bmh')
cwd = os.getcwd()

home = os.path.expanduser("~")
style.use(home+'\\.matplotlib\\stylelib\\fountain.mplstyle')

px = 6.45e-6 / 1.56

def n1d(y,wr,wy,lT,kT,q0):
    n = (2.*pi/(m6Li*wr**2))*(kT/lT**3) * utility.get_fp(q0-0.5/kT*m6Li*(wy*y)**2)
    return n
    
def secMom(y,n):
    sm = np.sum(y**2*n)/np.sum(n)
    return sm

def harmonic_trap_nOD(y,kT,mu,wr,wy):
    lT = np.sqrt(h**2/(2*pi*m6Li*kT))
    n = n1d(y_sample,wr,wy,lT,kT,mu0/kT) ## atoms line density
    Ntot = np.sum(n)*px
    nOD = n* sigma0 ## from atom density to optical density
    sm = secMom(y_sample/px,nOD) ## in px**2
    T_sm,ttf = utility.get_T_Unitary(wx,wy,wz,U0,sm,Ntot)
    return nOD, T_sm, sm, Ntot,ttf
    
close('all')
fig,ax = subplots(1)
figT,axT = subplots(3,1,sharex=True,figsize=(3,6))

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
height = 200
y_sample = np.arange(-height/2*px,height/2*px,px)

mu0 = 0.3e-6*kb

df = pd.DataFrame(columns=['Temp','sm','N','nOD','EE0','EF','ttf'])

kT_list = np.arange(50,500,5)*1e-9*kb
T_list = np.zeros(len(kT_list))
sm_list = np.zeros(len(kT_list))
N_list = np.zeros(len(kT_list))
N_list = np.zeros(len(kT_list))
N_list = np.zeros(len(kT_list))
nOD_list = np.zeros([len(kT_list),len(y_sample)])

for i,kT in enumerate(kT_list):
    nOD, T_sm, sm, Ntot,ttf = harmonic_trap_nOD(y_sample,kT,mu0,wr,wy)
    EE0,Ef = analysis.calc_EoverE0(wx,wy,wz,U0,sm,Ntot)
##     ax.plot(y_sample,nOD)
    ##Start second moment analysis
    df.loc[i] = [kT/kb*1e9,sm,Ntot,nOD,EE0,Ef,ttf]
    T_list[i] = T_sm
    sm_list[i] = sm
    N_list[i] = Ntot
    nOD_list[i] = nOD
    
    
axT[0].plot(kT_list/kb*1e9,T_list/kT_list*kb)
axT[0].set_ylim(0,2)

axT[1].plot(kT_list/kb*1e9,sm_list)
axT[2].plot(kT_list/kb*1e9,N_list)
axT[0].set_xlabel(r'Temperature (nK)')

ax.imshow(nOD_list)

ylabs = {   r'$\frac{T_\mathrm{sm}}{T}$',
            r'second moment',
            r'N'
        }

for my_ax,lab in zip(axT,ylabs):
    my_ax.set_ylabel(lab)
## reps = 100
## noise = np.array([np.random.normal(0,1,shape(n2d)) for i in range(reps)])
## n2d_noise = np.zeros(shape(noise))
## noise_amp = 0.1*(np.amax(n2d) - np.amin(n2d))
## noise *= noise_amp
## n2d_noise = n2d + noise
## n1d_noise = np.sum(n2d_noise,axis=2)
## im=ax[0].imshow(n2d,cmap='Greys_r')
## im2=ax[1].imshow(n2d_noise[0],cmap='Greys_r')
## im2.set_clim(im.get_clim())
fig.subplots_adjust(right=0.8)
figT.tight_layout()
lab = 'mu_'+str(int(mu0/kb*1e9))+'_T_' + str(int(kT/kb*1e9))
fig.savefig('Cloud_' + lab + '.pdf')
df.to_pickle(lab+'.pkl')
close('all')

#########################
##      Start PCA analysis here
#########################
my_pca = PCA_decomposition(df.nOD)
my_pca.get_eigensystem()
show()