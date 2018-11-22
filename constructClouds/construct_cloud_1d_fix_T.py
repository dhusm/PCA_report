import numpy, pickle

from pylab import *
from qo.theory import dipolepotentials
from qo.constants import *
from qo.evaltools import analysis
import pandas as pd
import PCA

from optparse import OptionParser
import csv, sys, copy, os
import scipy.optimize as opt
import utility
from scipy.stats import moment

## style.use('bmh')
cwd = os.getcwd()

home = os.path.expanduser("~")
style.use('C:\\Users\\d_hus\\.matplotlib\\stylelib\\fountain.mplstyle')

px = 6.45e-6 / 1.56
height = 150
y_sample = np.arange(0,height/2*px,px)

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
    Ntot = np.sum(n)*px*2
    nOD = n* sigma0 ## from atom density to optical density
    sm = secMom(y_sample/px,nOD)*2 ## in px**2
    T_sm,ttf = utility.get_T_Unitary(wx,wy,wz,U0,sm,Ntot)
    return nOD, T_sm, sm, Ntot,ttf
    
def get_N(mu,y,kT,wr,wy):
    lT = np.sqrt(h**2/(2*pi*m6Li*kT))
    n = n1d(y_sample,wr,wy,lT,kT,mu/kT) ## atoms line density
    Ntot = np.sum(n)*px*2
    return Ntot
    
def find_mu(y,kT,Ntot,wr,wy):
    f_opt = lambda mu: np.abs(get_N(mu*1e-9*kb,y_sample,kT,wr,wy) - Ntot)
    popt = opt.minimize_scalar(f_opt)
    mu = popt.x*1e-9*kb
##     print popt.fun
    return mu
    
    
close('all')
my_dict = dict(left=0.15,right=0.8)
fig = figure(figsize=(3,3))
axL = fig.add_axes([0.15, 0.73, 0.6, 0.23])
ax = fig.add_axes([0.15, 0.15, 0.6, 0.5],sharex=axL)
figT,axT = subplots(1,1,sharex=True,figsize=(3,3))

nuFortPower = 0.8 ##During imaging
nuFort=dipolepotentials.Dipoletrap(waistx=64.7e-6,waisty=80.e-6,wavelength=1064e-9,power=nuFortPower)	
trapFreqs=nuFort.trap_frequencies(dipolepotentials.Li6)
wx = trapFreqs[0]*2*pi
wz = trapFreqs[1]*2*pi
wr = np.sqrt(wx*wz)
wy = 31.1*2*pi
wxyz = np.array([wx,wy,wz])
U0 = -nuFort.potential(dipolepotentials.Li6)
sigma0 = 0.5*3*0.67e-6**2/2/pi

SNR = 0.1

df = pd.DataFrame(columns=['Temp','sm','sm0','mu','nOD','Ntot','EE0','EF','ttf'])

kT = 100e-9*kb
Ntot_list = np.arange(10e3,200e3,5e3)
tag = 'T_'+str(int(kT/kb*1e9)) + 'k_' + 'SNR_' + str(SNR) + '_'
for i,Ntot in enumerate(Ntot_list):
    
    mu = find_mu(y_sample,kT,Ntot,wr,wy)
    _, _, sm0, _,_ = harmonic_trap_nOD(y_sample,kT,mu,wr,wy,noise=0)
    nOD, T_sm, sm, Ntot,ttf = harmonic_trap_nOD(y_sample,kT,mu,wr,wy,noise=SNR)
    EE0,Ef = analysis.calc_EoverE0(wx,wy,wz,U0,sm,Ntot)
    
    df.loc[i] = [kT/kb*1e9,sm,sm0,mu,np.array(nOD,dtype='float'),Ntot,EE0,Ef,ttf]
    

nOD_list = np.array([row for row in df.nOD.values])
my_nOD = nOD_list[-1]
la = [5,15,25,35]
X,Y = np.meshgrid(y_sample*1e6,Ntot_list*1e-3)
im = ax.pcolormesh(X,Y,nOD_list/sigma0*1e-6,cmap=cm.gray)
for l in la:
    ax.plot(y_sample*1e6,np.ones(len(y_sample))*Ntot_list[l]*1e-3,'r-')
axL.plot(y_sample*1e6,nOD_list[la].T/sigma0*1e-6,'k-')
axL.set_ylabel(r'$n_{\mathrm{1D}} (\mathrm{atoms}/\mu\mathrm{m})$')
## axL.set_ylim([-20,500])
ax.set_xlabel(r'$y$ ($\mu$m)')
ax.set_ylabel(r'$N$ ($10^3$)')
axL.set_xlim([0,300])
## axL.set_xticklabels([])
## fig.tight_layout()
## im_aa2.set_clim([-0.4,0.4])
bbox = ax.get_position().extents
cbaxes = fig.add_axes([0.82, bbox[1], 0.03, bbox[3]-bbox[1]]) 
cbar = colorbar(im,cax=cbaxes)
## cbar = colorbar(im,ax=[ax,axL])
cbar.set_label(r'$n_{\mathrm{1D}} (\mathrm{atoms}/\mu\mathrm{m})$')
fig.savefig(tag+'1D_density.png',dpi=300)


#########################
##      Start PCA analysis here
#########################
my_pca = PCA.PCA_decomposition(df.nOD)
my_pca.get_eigensystem()

f1,a1 = subplots(1,figsize=(3,3))
a1.plot(my_pca.val,'o',ms=3)
a1.set_yscale('log')
a1.set_xlabel('Rank of PC')
a1.set_ylabel('Eigenvalue of PC')
f1.tight_layout()
f1.savefig(tag+'PC_eigenvalues.pdf')

vecs = np.array([vec for vec in my_pca.vec])
f2, [a2,aa2] = subplots(2,figsize=(3,4))
X,Y = np.meshgrid(y_sample*1e6,range(len(vecs.T)))
## im_a2 = a2.imshow(vecs,cmap=cm.RdBu)
im_a2 = a2.pcolormesh(X,Y,vecs.T,cmap=cm.RdBu)
im_a2.set_clim([-0.4,0.4])
a2.set_ylabel('PC rank')
colorbar(im_a2,ax=a2)
nl = 10
X,Y = np.meshgrid(y_sample*1e6,range(nl))
im_aa2 = aa2.pcolormesh(X,Y,vecs.T[:nl],cmap=cm.RdBu)
im_aa2.set_clim([-0.4,0.4])
aa2.set_ylabel('PC rank')
aa2.set_xlabel(r'$y$ ($\mu$m)')
colorbar(im_aa2,ax=aa2)
f2.tight_layout()
f2.savefig(tag+'Eigenvectors.png',dpi=300)

sm_rec = np.zeros([len(Ntot_list),len(my_pca.val)])
## Reconstruct clouds from various numbers of PCA
for n,nOD in enumerate(nOD_list):
    for i in range(len(my_pca.val)):
        rec_pic = my_pca.reconstruct_from_PCA(nOD,i)* sigma0
        sm = np.sum((y_sample/px)**2*rec_pic)/np.sum(rec_pic)*2
        sm_rec[n,i] = sm
sm_dif = sm_rec.T / df.sm.values - 1
m = 50
axT.plot(sm_dif[:m,::10]*100,'-o',ms=3)
labs = [str(int(it*1e-3)) for it in Ntot_list[::10]]
## im = axT.imshow(sm_dif[:10].T,aspect='auto',cmap=cm.RdBu)
## im.set_clim([-0.1,0.1])
## colorbar(im,ax=axT)
axT.set_ylim([-10,10])
axT.set_xlabel('Number of considered PC')
axT.set_ylabel('deviation from origianl second moment (\%)')
axT.legend(labs,frameon=False)
figT.tight_layout()
figT.savefig(tag+'deviation.pdf')
show()