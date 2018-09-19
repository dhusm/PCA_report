# -*- coding: utf-8 -*-

import numpy

class CamSettings(object):
  
  def __init__(self):
    self.bin =			[1,1]
    self.pixelsize =		6.45/1.55
    self.camaxis =			"X"
    self.Csat_per_time =	None
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
		self.sigma0 = None
		localName   = ""
		remoteZName = ""
		remoteXName = ""
