##########################################
# Template for analysing transport data from X-Cam
# Only processing, no loading and reading. But you can use the data of several runs at once
#
# Version from 2014 - 09 - 16
# created by S. Krinner
##########################################

import numpy, pickle
from pylab import *
import qo.evaltools.loader_new  as loader
import qo.evaltools.reader_fast as reader
from qo.evaltools import analysis
from qo.theory import dipolepotentials
from qo.constants import *
import scipy.optimize as opt
import settings
import pandas as pd
from PCA import PCA_decomposition
import sys, os
import utility

#########################################
## create experiment settings
#########################################
#%%
# Informations about the run to be loaded : name, place where to store the resulting file, and location of the picture files on the camera computers
day = '22'
month = '11'
year = '2016'

############################################
## Trap Parameter
############################################
nuFortPower=0.4	# power in Watt at transport
nuFortPowerIm=0.8	# power in Watt at imaging
wx,wy,wz,w_bar,U0 = utility.getFortFreq(nuFortPower)

firstFlag = 1
cwd=os.getcwd()
exper = settings.ExpSettings() 
exper.localName   = cwd + '\\'
path_name = year+month+'\\'+year+month+day+'\\'
exper.remoteZName = 'Z:\\' + path_name
exper.remoteXName = 'X:\\' + path_name

#########################################
## Runs to load, options for exclusion
#########################################
runNames=['fringes_Xcam_atoms_Run1']
          
# put parameters of interest into this list
parameterList=['i']

#########################################
## camera settings
#########################################
cam = settings.CamSettings()
## define regions of interests (ROI), separately for fitting and counting
## for counting you want a small region to minimize shot noise from the background
# Center coordinates of the cloud in the full pictures
y0 = 304
x0 = 1200
height = 120 # actually half the height
width = 40 # actually half the width

param_dict = {"removeCondPad":True, "excludeList":[],"countRuns":10000}
cam.ROI_reader 	= numpy.array([y0-height,y0+height,x0-width,x0+width])
cam.noAtomsROI = numpy.array([230,370,1100,1145])# area used to get sFactor

cam.setRunNamesDict(runNamesList=runNames,singleRunParams=param_dict)
cam.setLogNamesDict(cam.runNamesDict)
cam.setRootNamesDict(cam.runNamesDict)
#%%
if firstFlag:
	for runName,logName in zip(cam.runNamesDict,cam.logNamesDict):
		print(' ')
		print("******** Running the loader, please wait... *********")
		print(' ')
		a = loader.CameraLoad(runName,exper.localName,exper.remoteZName,exper.remoteXName,cam.runNamesDict[runName]["excludeList"],cam.runNamesDict[runName]["countRuns"],logName, rm_condPad = cam.runNamesDict[runName]["removeCondPad"])
		a.loadXFileNames()
		a.makeNewLog()
  
	for logName,rootName in zip(cam.logNamesDict,cam.rootNamesDict):
		print(' ')
		print("******** Running the reader, please wait... *********")
		print(' ')
		a = reader.CameraRead(logName,exper.localName,binSize=cam.bin,ROI = cam.ROI_reader/cam.bin[0],noXAtomsROI = cam.noAtomsROI, Csat_per_time=cam.Csat_per_time,CameraAxis=cam.camaxis)
		a.getRuns(rootName)
		a.makeNewLog()


ROI_offset=[0,240,25,50] 	# ROI to determine the offset from linesum along y (=zeroth) coordinate #[0,240, 0,120]
ROI_dip=[50,184,20,54]		# ROI to count atom number Ntot, on the complement a plane is fitted to subtract offset and slope
ROI_count=[30,194,20,60]	# ROI to count atom number Nup, Ndown, 
ROI_sm=[0,240,0,80]  		# ROI for second moment calculation

lightsheet_position=125 #Position of light sheet determines boundaries between upper and lower reservoir

############################################
#Do actual processing
############################################
#Load  all the runs and add the relevant data to a single array.
#Postselection is done on the individual run level
# flags
forceAnalysis = 1
forceflag=1
forceflagDF=1
forceflagUnitary=1

AnalyzerDict={} #Dictionary that contains the different analyzer objects and their logs
resDict={}
print(' ')
print("******** Running the data analysis *********")
print(' ')
for logName in runNames:
    AnalyzerDict[logName] = analysis.Analyzer(logName + 'preProcessed', exper.localName, ROI = ROI_offset)
    if forceAnalysis:
        AnalyzerDict[logName].createPCAPictures(ROI_sm,n=10,force=0,doAll=False)
        AnalyzerDict[logName].makePictureArray('XDensityPCA',force=False)
        AnalyzerDict[logName].makePictureArray('XDensityHighSat',force=False)
#        AnalyzerDict[logName].add_parameters_to_log(parameterList, force=False)	# add parameters of interest to the log-file
#    
#        AnalyzerDict[logName].get_Ntot(ROI_dip,picName='XDensityHighSat', force=forceflag) # Count total atom number
#        AnalyzerDict[logName].get_Nup_Ndown(lightsheet_position,picName='XDensityHighSat', ROI_dip=ROI_dip, ROI_count=ROI_count, force=forceflag) # Count atom number in upper and lower reservoir
#        AnalyzerDict[logName].get_deltaN(force=forceflag)	# compute deltaN based on the values of Nup and Ndown
#    #
#    #    AnalyzerDict[logName].get_secondMomentUp_secondMomentDown(lightsheet_position,ROI_offset,ROI_sm,force=forceflagUnitary)
#        AnalyzerDict[logName].get_secondMomentUp_secondMomentDown(lightsheet_position,
#        ROI_offset,ROI_sm,force=forceflagUnitary,pic='XDensityPCA',crop_barrier=2,x_offset=-1.5)
#        AnalyzerDict[logName].get_Tup_Tdown_unitary(wx,wy,wz,U0,force=forceflagUnitary)	# needs secondMoment_up, secondMoment_down, TupDF, TdownDF!
#        AnalyzerDict[logName].postSelect(parameterName="N_from_XDensityHighSat", threshold=3., absval=None, fromnothing=True)
#        
#    resDict[logName]=AnalyzerDict[logName].get_values([
#							parameterList[0],
#                                        parameterList[1],
#							"N_from_XDensityHighSat",
#							"Nup",
#							"Ndown",
#							"deltaN",
#							"secondMoment_up",
#							"secondMoment_down",
#							"Tup_unitary",
#							"Tdown_unitary",
#                                        "EoverEf_up_unitary",
#                                        "EoverEf_down_unitary",
#                                        "SoverNkb_up_unitary",
#                                        "SoverNkb_down_unitary"
#							])
figure()
#%%
#close('all')
fn = os.listdir(os.getcwd() + '\\data')[-1]
f = open(os.getcwd() + '\\data\\' + fn)
data = pickle.load(f)
f.close()
pic = data['pictures']
ROI = ROI_sm
dens = pic['XDensityPCA']
dens_hs = pic['XDensityHighSat']
dens_orig = pic['XDensityHighSat'][ROI[0]:ROI[1], ROI[2]:ROI[3]] / exper.sigma0
plot(np.sum(dens_orig,axis=1))
plot(np.sum(dens,axis=1))
#%%
od = dens#[ROI[0]:ROI[1],ROI[2]:ROI[3]] / exper.sigma0
im = od[ROI[0]:ROI[1], ROI[2]:ROI[3]]
imup = od[ROI[0]:lightsheet_position, ROI[2]:ROI[3]]
imdown = od[lightsheet_position:ROI[1], ROI[2]:ROI[3]]
