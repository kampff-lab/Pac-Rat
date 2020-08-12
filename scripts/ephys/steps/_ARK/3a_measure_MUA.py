# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 2a: measure MUA

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)

# Specify session folder
#session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

# Specify data paths
data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')

# Detect spikes on each channel (using weak thresholds for picking up many spikes)
ephys.detect_MUA(data_path)

# To Do...
# - Detect artifact spikes (when spikes occur on too many channels within 1 ms of one another)
# - Count spikes per video-frame (removing or setting to NaN frames with artifact spikes)
# -  - this "spike count" per frame per channel will be the MUA signal that we use for subsequent analysis 
