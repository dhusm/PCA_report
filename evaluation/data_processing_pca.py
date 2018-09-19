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
import qo.evaltools.loader as loader
import qo.evaltools.reader as reader
from qo.expwiz.control_basics import Formula
from qo.expwiz.imagefile import ImageFile
from qo.expwiz.fits import FitFermi, FitCount
from qo.expwiz import config
from qo.evaltools import analysis
from qo.theory import gsl, dipolepotentials
from qo.constants import *
import settings
import pandas as pd
import sys
import PCA
## from PCA import PCA_decomposition

from optparse import OptionParser
import csv, sys, copy, os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.optimize as opt
from matplotlib import rc
import utility


#########################################
## create experiment settings
#########################################

cwd=os.getcwd()
exper = settings.ExpSettings() 
exper.localName   = os.getcwd() + '\\'
exper.remoteZName = "Z:\\201609\\20160926\\"
exper.remoteXName = "X:\\201609\\20160926\\"
exper.sigma0 = 0.5*3*0.67**2/2./pi# !! be careful: I use mu**2 not m**2 !! And: factor 0.5 for x-imaging (orthogonal to magnetic field, unlike for z-imaging where it is parallel)

# Informations about the run to be loaded : name, place where to store the resulting file, and location of the picture files on the camera computers

runNames=["calibration_MeasureTemperatures_PNFE400_Run1"]

exclude_list = []
x_pics = ['XGlassAtoms','XGlassBright','XGlassDark','XGlassDensity_down','XGlassDensity_up','XGlassDensity','XGlassRaw']

param_dict = {"removeCondPad":True, "excludeList":exclude_list,"countRuns":10000}
cam = settings.CamSettings()
cam.bin = numpy.array([1,1])
cam.pixelsize = 6.45/1.56	# in micron, divided by magnification
cam.camaxis = "X"		# give X or Z for cam axis
cam.Csat_per_time = 156	# in units 1/[mus]. At the moment not independent of binning, 150 is value for (1,1) binning

## define regions of interests (ROI), separately for fitting and counting
## for counting you want a small region to minimize shot noise from the background
cam.ROI_reader 	= numpy.array([154,454,1160,1240])
cam.ROI_analysis = numpy.array([64,234,15,65])		# choose bigger area than for counts
cam.noAtomsROI = numpy.array([230,370,1100,1145])	# area used to get sFactor
wallpos = 90
counts_crop = numpy.array([-20,20,-10,10])			# choose crop region as cut off border of ROI_analysis for fits
cam.ROI_counts	= cam.ROI_analysis - counts_crop		# choose narrow area, including shot noise background can create significant errors
wallpos_counts = wallpos + counts_crop[0]
xi = 0.37

cam.setRunNamesDict(runNamesList=runNames,singleRunParams=param_dict)
cam.setLogNamesDict(cam.runNamesDict)
cam.setRootNamesDict(cam.runNamesDict)

loadFlag = False
readFlag = False

if loadFlag:
	for runName,logName in zip(cam.runNamesDict,cam.logNamesDict):
		print(' ')
		print(' ')
		print("******** Running the loader, please wait... *********")
		print(' ')
		print(' ')
		a = loader.CameraLoad(runName,exper.localName,exper.remoteZName,exper.remoteXName,cam.runNamesDict[runName]["excludeList"],cam.runNamesDict[runName]["countRuns"],logName, rm_condPad = cam.runNamesDict[runName]["removeCondPad"])
		a.loadXFileNames()
		a.makeNewLog()

###########################################

if readFlag:
	for logName,rootName in zip(cam.logNamesDict,cam.rootNamesDict):
		print(' ')
		print(' ')
		print("******** Running the reader, please wait... *********")
		print(' ')
		print(' ')
		a = reader.CameraRead(logName,exper.localName,binSize=cam.bin,ROI = cam.ROI_reader/cam.bin[0],noXAtomsROI = cam.noAtomsROI, Csat_per_time=cam.Csat_per_time,CameraAxis=cam.camaxis)
		a.getRuns(rootName)
		a.makeNewLog()


############################################
## Parameters for processing
############################################
# trapping frequencies degenerate Fermi fits
nuFortPower=0.2	# power in Watt
nuFortPowerIm=0.8	# power in Watt
nuFort=dipolepotentials.Dipoletrap(waistx=64.7e-6,waisty=80.e-6,wavelength=1064e-9,power=nuFortPower)	#waists chosen such that for a power of 0.150W, nux=194Hz and nuz=157Hz, as measured for the quantized conductance paper
trapFreqs=nuFort.trap_frequencies(dipolepotentials.Li6)
omegax= 2*pi*trapFreqs[0]
omegaz= 2*pi*trapFreqs[1]
omegay= 2*pi*31.1				# determined by Feshbach field, WIF: 23.5Hz, Unitary: 31.1Hz
omega_bar=(omegax*omegay*omegaz)**(1/3.)
U0=-nuFort.potential(dipolepotentials.Li6)	# trap depth
nu_x = 10e3
nu_z = 10e3

ROI_offset=[0,300,25,50] 	# ROI to determine the offset from linesum along y (=zeroth) coordinate #[0,240, 0,120]
ROI_dip=[80,214,20,54]		# ROI to count atom number Ntot, on the complement a plane is fitted to subtract offset and slope
ROI_count=[80,214,20,54]	# ROI to count atom number Nup, Ndown, 
ROI_sm=[60,234,20,54]  		# ROI for second moment calculation
ROI_DF=[64,234,15,65]		# ROI for degenerate Fermi fit
ROI_1dfit=[0,300,10,70]		# ROI for 1d Fermi fit

lightsheet_position=154 #Position of light sheet determines boundaries between upper and lower reservoir

# put parameters of interest into this list, each value has to be passed as argument to the get_values method below if the parameter should appear in the recored array
parameterList=["P_Dimple_Mod_Amp"]	# Transport_Time "TOF_2_Time", "Dimple_modamp", "I_CloverleafY_Offset","P_FORT_Final_End"

##########################################
# PCA
##########################################
pcaLoadFlag = 0
pcaDiagFlag = 1
pcaSaveFlag = 1
pic_path = cwd+'\\data\\'
ROI_pca = [0,300,20,60]
ROI_pca_sm_offset = [0,300,0,40]
ROI_pca_sm = [0,300,0,40]
ROI_pca_sm_offset_sep = [0,140,0,40]
ROI_pca_sm_sep = [0,140,0,40]
ls_pca = 74

print(' ')
print("******** Apply Principle Component Analysis *********")
print(' ')

pic_num = len(os.listdir(pic_path)) # Number of pictures in data folder
pic_list = np.zeros([pic_num,ROI_pca[1]-ROI_pca[0],ROI_pca[3]-ROI_pca[2]])
pic_name_list = [] # store here the file names of pictures with are not lost shots
print 'Loading the pictures from: ', exper.remoteXName
print 'This could take a while, there are ',str(len(os.listdir(pic_path))), ' pictures to load!'
i = 0 # Good shots
j = 0 # Bad shots
if pcaLoadFlag:
    for fname in os.listdir(pic_path):
        with open(pic_path + fname) as f:
            data = pickle.load(f)
        p = (data['pictures']['XDensityHighSat']) / exper.sigma0
        p = p[ROI_pca[0]:ROI_pca[1],ROI_pca[2]:ROI_pca[3]]
        if np.mean(p)>0.1:
            pic_list[i,:,:] = p
            pic_name_list.append(fname)
            i += 1
            sys.stdout.write("*")
            sys.stdout.flush()
        else:
            j += 1
            sys.stdout.write("X")
            sys.stdout.flush()
    print '\nExcluded ', str(j), ' pictures from the PCA.' 
    with open('full_pic_list.pkl','wb') as f:
        pic_list = pic_list[:i,:,:]
        pickle.dump((pic_list,pic_name_list),f)
else:
    with open('full_pic_list.pkl','rb') as f:
        pic_list,pic_name_list = pickle.load(f)

cutoff = 1000     
if pcaDiagFlag:
    my_pca = PCA.PCA_decomposition(pic_list[:cutoff])
    my_pca_up = PCA.PCA_decomposition(pic_list[:cutoff,80:lightsheet_position,:])
    my_pca_down = PCA.PCA_decomposition(pic_list[:cutoff,lightsheet_position:-80,:])
    val,vec = my_pca.get_eigensystem(doReduced=True)
    val_up,vec_up = my_pca_up.get_eigensystem(doReduced=True)
    val_down,vec_down = my_pca_down.get_eigensystem(doReduced=True)
    with open('EigenAnalysis.pkl','wb') as f:
        pickle.dump(my_pca,f)
    with open('EigenAnalysis_up.pkl','wb') as f:
        pickle.dump(my_pca_up,f)
    with open('EigenAnalysis_down.pkl','wb') as f:
        pickle.dump(my_pca_down,f)        
else:
    with open('EigenAnalysis.pkl','rb') as f:
        my_pca = pickle.load(f)
    with open('EigenAnalysis_up.pkl','rb') as f:
        my_pca_up = pickle.load(f)
    with open('EigenAnalysis_down.pkl','rb') as f:
        my_pca_down = pickle.load(f)
#%%
n_list = np.arange(100)
#n_list = np.array([2,100])
cut = 5
pic_list = pic_list[:cut]
pic_list_up = pic_list[:,80:lightsheet_position,:]
pic_list_down = pic_list[:,lightsheet_position:-80,:]
sm = np.zeros([len(n_list),len(pic_list),2])
sm_single = np.zeros([len(n_list),2])
sm_single_sep = np.zeros([len(n_list),2])

#%%
my_pca.plot_PCA(5,save=True,ylims=[50,250])
my_pca_up.plot_PCA(5,save=True,fname='PC_up')
my_pca_down.plot_PCA(5,save=True,fname='PC_down')
pic_old = np.zeros(shape(pic_list[0]))
pic_new = np.zeros(shape(pic_list[0]))
var_stored = np.zeros(len(n_list))
#fig,ax=subplots(1,3)
#ax[0].imshow(pic_list[0],cmap='Greys_r')
dummy = True
for n in n_list:
    # keep the largest n principal components
    print '\nReconstructing the pictures from the ',str(n), ' largest principal components...'
    sys.stdout.flush()
    pic_old = pic_new
    rec_pic = [my_pca.reconstruct_from_PCA(pic,n) for pic in pic_list]
#    rec_pic_up = [my_pca_up.reconstruct_from_PCA(pic,n) for pic in pic_list_up]
#    rec_pic_down = [my_pca_down.reconstruct_from_PCA(pic,n) for pic in pic_list_down]
#    rec_pic_glue = np.concatenate([rec_pic_up,rec_pic_down],axis=1)
#    pic_new = rec_pic_glue[0]
    sys.stdout.write("*")
    sys.stdout.flush()
    sm_single[n] = analysis.calc_secondMomentUp_secondMomentDown_1d(rec_pic[0],lightsheet_position,
            ROI_pca_sm_offset,ROI_pca_sm,crop_barrier=2,forcePlot=False)
#    sm_single_sep[n] = analysis.calc_secondMomentUp_secondMomentDown_1d(rec_pic_glue[0],lightsheet_position-80,ROI_pca_sm_offset_sep,ROI_pca_sm_sep)
#    my_var = np.sum(np.abs(pic_old - pic_new))
#    var_stored[n] = my_var
#    sm[n] = [analysis.calc_secondMomentUp_secondMomentDown_1d(rec_pic_glue[fc],lightsheet_position,
#            ROI_pca_sm_offset,ROI_pca_sm) for fc in range(len(rec_pic_glue))]
#ax[0].set_xticks([0,40])
#ax[1].set_xticks([0,40])
#ax[0].set_ylim([50,250])
#ax[1].set_ylim([50,250])
#ax[1].get_yaxis().set_visible(False)
#ax[2].get_yaxis().set_visible(False)
#ax[2].set_ylim([50,250])
#fig.subplots_adjust(wspace=0)
#fig.tight_layout()

#%%
fig,ax=subplots(2,sharex=True)
#ax_down = ax[0].twinx()
#ax[0].plot(n_list,sm[:,:,0],'r')
#ax_down.plot(n_list,sm[:,:,1],'b')
ax[0].plot(n_list,sm_single[:,0],'r',label='up')
ax[0].plot(n_list,sm_single[:,1],'b',label='down')
#ax[0].plot(n_list,sm_single_sep[:,0],'r--',label='up')
#ax[0].plot(n_list,sm_single_sep[:,1],'b--',label='down')
#legend()
ax[0].set_xlabel('Number of principal components')
ax[0].set_ylabel('Second Moment')
fig.savefig('sm_vs_n.pdf')
#%%
"""
############################################
#Do actual processing
############################################
#Load  all the runs and add the relevant data to a single array.
#Postselection is done on the individual run level

# flags
forceflag=False
forceflagDF=False
forceflagUnitary=False

AnalyzerDict={} #Dictionary that contains the different analyzer objects and their logs
resDict={}
for logName in runNames:
	AnalyzerDict[logName] = analysis.Analyzer(logName + 'preProcessed', exper.localName, ROI = ROI_offset)
	AnalyzerDict[logName].add_parameters_to_log(parameterList, force=False)	# add parameters of interest to the log-file

	AnalyzerDict[logName].get_Ntot(ROI_dip, force=forceflag) # Count total atom number
	AnalyzerDict[logName].get_Nup_Ndown(lightsheet_position, ROI_dip=ROI_dip, ROI_count=ROI_count, force=forceflag) # Count atom number in upper and lower reservoir
	AnalyzerDict[logName].get_deltaN(force=forceflag)	# compute deltaN based on the values of Nup and Ndown

#	AnalyzerDict[logName].get_Tup_Tdown_degFermi(ROI_DF, ROI_dip, omegax, omegay, lightsheet_position, force=forceflagDF)
#	AnalyzerDict[logName].get_T_degFermi(ROI_DF, ROI_dip, omegax, omegay, force=forceflagDF)
#	AnalyzerDict[logName].get_T_degFermi_1d(ROI_DF, ROI_dip,force=forceflagDF)

#	AnalyzerDict[logName].get_secondMomentUp_secondMomentDown(lightsheet_position,ROI_offset,ROI_sm,force=forceflagUnitary)
    	sm = AnalyzerDict[logName].get_secondMomentUp_secondMomentDown(lightsheet_position,ROI_pca_sm_offset,ROI_pca_sm,force=forceflagUnitary,pic='XDensityPCA')
	AnalyzerDict[logName].get_Tup_Tdown_unitary(omegax,omegay,omegaz,U0,force=forceflagUnitary)	# needs secondMoment_up, secondMoment_down, TupDF, TdownDF!
## 	AnalyzerDict[logName].get_secondMoment(ROI_offset,ROI_sm,force=forceflagUnitary)
	#~ AnalyzerDict[logName].get_T_unitary(omegax,omegay,omegaz,U0,force=forceflagUnitary)	 # needs secondMoment, TDF!

	AnalyzerDict[logName].postSelect(parameterName="N_from_XDensityHighSat", threshold=3., absval=None, fromnothing=True)
	resDict[logName]=AnalyzerDict[logName].get_values([
							parameterList[0],
							"N_from_XDensityHighSat",
							"Nup",
							"Ndown",
							"deltaN",
#							"NupDF",
#							"NdownDF",
#							"TupDF",
#							"TdownDF",
#							"MuupDF",
#							"MudownDF",
#							"TupyDF",
#							"TdownyDF",
#							"T_DF",
#							"Mu_DF",
#							"Nfit_DF",
#							"T_DF_y",
							"secondMoment_up",
							"secondMoment_down",
							"secondMoment",
							"Tup_unitary",
							"Tdown_unitary"
							#~ "T_unitary"
							])
    
    

close('all')

fn = os.listdir(os.getcwd() + '\\data')[0]
f = open(os.getcwd() + '\\data\\' + fn)
data = pickle.load(f)
f.close()
pic = data['pictures']
dens = pic['ComputedXDensity']
ROI = ROI_DF
od = dens#[ROI[0]:ROI[1],ROI[2]:ROI[3]] / exper.sigma0
im = od[ROI[0]:ROI[1], ROI[2]:ROI[3]]
imup = od[ROI[0]:lightsheet_position, ROI[2]:ROI[3]]
imdown = od[lightsheet_position:ROI[1], ROI[2]:ROI[3]]

df = [pd.DataFrame(resDict[name]) for name in runNames]

for dfs in df:
	dfs.ix[dfs.N_from_XDensityHighSat < 40000, 'N_from_XDensityHighSat'] = np.NaN
df0 = df[0].set_index(parameterList[0])
for dfs in df[1:]:
	dfs = dfs.set_index([parameterList[0]])
	df0 = df0.combine_first(dfs)
 
resmean = df0.mean(level=(parameterList[0]))
resstd = df0.std(level=(parameterList[0]))
df = resmean.unstack(parameterList[0])
# set colorscale and create indexed list to set color for each run
idx = 1-np.linspace(0.,1,len(runNames))

# Perform average over shots having the same "parameterList[0]" value, preSorting after spin or whatever can be done when setting preSort=True and specifying presortparameterName
#    resmean, resstd = AnalyzerDict[name].perform_average(resDict[name],parameterList[0], preSort=False)
Ntot = resmean['N_from_XDensityHighSat']
Ntot_std = resstd['N_from_XDensityHighSat']
Nup = resmean["Nup"]
Ndown = resmean["Ndown"]
relimba = resmean["deltaN"]
relimba_std = resstd["deltaN"]
Tup = resmean["Tup_unitary"]
Tup_std = resstd["Tup_unitary"]
Tdown = resmean["Tdown_unitary"]
Tdown_std = resstd["Tdown_unitary"]
sm_up = resmean["secondMoment_up"]
sm_up_std = resstd["secondMoment_up"]
sm_down = resmean["secondMoment_down"]
sm_down_std = resstd["secondMoment_down"]
Tup = resmean["Tup_unitary"]
scanValues = Ntot.index.values # Transport_Time array for x axis
	
###################################
## TEMPERATURE, average over one decay curve
###################################
Tup_run = np.average(Tup)
Tdown_run = np.average(Tdown)
trapHeating = np.mean(2e-9 * (4.458 - scanValues / 2.)) # Heating from waiting in the dipole trap. Take average over all transport times.
N = np.mean(Ntot)
T_raw = np.mean([Tup_run,Tdown_run]) # Temperature from FermiFits, avg over the two reservoirs
Ef =hbar * omega_bar * (6*N)**(1/3.)
Tf_up = hbar * omega_bar * (6*Nup*2)**(1/3.) /kb
Tf_down = hbar * omega_bar * (6*Ndown*2)**(1/3.) /kb
Tup = Tup * (nuFortPower/nuFortPowerIm)**(2/3.)
Tdown = Tdown * (nuFortPower/nuFortPowerIm)**(2/3.)
Tup_std = Tup_std * (nuFortPower/nuFortPowerIm)**(2/3.)
Tdown_std = Tdown_std * (nuFortPower/nuFortPowerIm)**(2/3.)
T = T_raw * (nuFortPower/nuFortPowerIm)**(2/3.)
T = T_raw - trapHeating
Tf = Ef/kb
ToverTf = T/Tf
Tup = Tup - trapHeating
Tdown = Tdown - trapHeating
Tdelta = Tdown - Tup
Tdelta_std = sqrt(Tup_std**2+Tdown_std**2)

Tup_mean = np.mean(Tup)
Tdown_mean = np.mean(Tdown)
mu_up = utility.getMuFromZwierlein(Tup,Tf_up)         
mu_down = utility.getMuFromZwierlein(Tdown,Tf_down)
dmu = mu_up - mu_down      

dimple_mod_amp = df.index.values

dmu_dT_list = np.zeros([len(dimple_mod_amp),3])

## plot the atom number difference and temepratures in the reservoirs
fig,ax = subplots(4,sharex=True,figsize=(5,8))
ax_dmu = ax[0].twinx()
ax[0].set_ylabel('Atom number difference')
ax[1].set_ylabel('Absolute temperature (uK)')
ax[2].set_ylabel('Temperature difference (uK)')
ax[3].set_ylabel('second Moment')
ax[2].set_xlabel('Transport_Time (s)')
ax[0].errorbar(scanValues,relimba*Ntot,yerr=relimba_std*Ntot,color='green')
ax[1].errorbar(scanValues,Tup*1e9,yerr=Tup_std*1e9, color='blue')
ax[1].errorbar(scanValues,Tdown*1e9,yerr=Tdown_std*1e9, color='red')
ax[2].errorbar(scanValues,Tdelta*1e9,yerr=Tdelta_std*1e9)
ax[3].errorbar(scanValues,sm_up*1e9,yerr=sm_up_std*1e9)
ax[3].errorbar(scanValues,sm_down*1e9,yerr=sm_down_std*1e9)

## plot the chemical potential bias over the course of the transport
ax_dmu.plot(scanValues,dmu/kb*1e9,color='k')
 
fig.subplots_adjust(hspace=0)
fig.tight_layout()
fig.savefig('Transport_ICYS.pdf')  
close()    

#dmu_dT_list[i] = [key,np.mean(Tdelta[key][:5])*1e9,np.mean(dmu[key][-5:])/kb*1e9]

i += 1


    
    
""" 