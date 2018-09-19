# -*- coding: utf-8 -*-

import numpy

class CamSettings(object):
  
  def __init__(self):
    self.bin =			[1,1]
    self.pixelsize =		6.45/1.56
    self.camaxis =			"X"
    self.Csat_per_time =	156
    self.ROI_reader = 		numpy.array([])
    self.ROI_analysis = 		numpy.array([])
    self.ROI_counts =		numpy.array([])
    self.noAtomsROI = 		numpy.array([])
    self.runNamesDict = 	{}
    self.logNamesDict = 		{}
    self.rootNamesDict = 	{}
    
  def setRunNamesDict(self,runNamesList,singleRunParams):
    for runName in runNamesList:
      self.runNamesDict[runName] =  singleRunParams
    return self.runNamesDict
    
  def setLogNamesDict(self,logNamesDict):
    self.logNamesDict = logNamesDict
    
  def setRootNamesDict(self,rootNamesDict):
    self.rootNamesDict = rootNamesDict
    

class ExpSettings(object):

	def __init__(self):
		self.sigma0 = 0.5*3*0.67**2/2./numpy.pi
		localName   = ""
		remoteZName = ""
		remoteXName = ""
