# =================================================================================== #
# AUXILIAR CLASSES
# =================================================================================== #

import os
from datetime import datetime

# =================================================================================== #
# Class for handling directories paths
class handling_dir:

    # -----------------------------------------
    # Attributes
    _mainDir = ''            # Routine main directory 
    _inputDir = ''           # Input data directory
    _modelDir = ''           # GNN model directory
    _outputDir = ''          # Output data directory

    # -----------------------------------------
    # Constructor
    def __init__(self):
        type(self)._mainDir = os.getcwd()
        type(self)._inputDir = type(self)._mainDir+'\\InputData'
        type(self)._modelDir = type(self)._mainDir+'\\GNNmodel'
        self._dh = datetime.now()
        self._dh_text = (self._dh.year,self._dh.month,self._dh.day,
                         self._dh.hour,self._dh.minute,self._dh.second)
        os.makedirs('OutputData\\Result-'+str(self._dh_text))
        type(self)._outputDir = os.getcwd()+'\\OutputData\\Result-'+str(self._dh_text)

    # -----------------------------------------   
    # Method to access - routine main directory
    @classmethod        
    def get_mainDir(cls):
        return cls._mainDir
    
    # -----------------------------------------   
    # Method to access - input data directory
    @classmethod        
    def get_inputDir(cls):
        return cls._inputDir
    
    # -----------------------------------------   
    # Method to access - GNN model directory
    @classmethod        
    def get_modelDir(cls):
        return cls._modelDir

    # -----------------------------------------   
    # Method to access - output data directory
    @classmethod        
    def get_outputDir(cls):
        return cls._outputDir     