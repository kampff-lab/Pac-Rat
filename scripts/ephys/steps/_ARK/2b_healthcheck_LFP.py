# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 2b: health check analysis and plots for LFP signal (low-frequency)

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

# Ephys Constants
num_raw_channels = 128
bytes_per_sample = 2
raw_sample_rate = 30000

# Specify session folder
session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
#session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

# Specify downsampled data path
data_path = os.path.join(session_path +'/Amplifier_downsampled.bin')

# Measure file size (and number of samples)
bytes_per_sample = 16
statinfo = os.stat(data_path)
num_samples = np.int(statinfo.st_size / bytes_per_sample)
num_samples_per_channel = np.int(num_samples / num_raw_channels)

# Memory map amplifier data
tmp = np.memmap(data_path, dtype=np.uint16, mode = 'r')
data = np.reshape(tmp,(num_samples_per_channel,128)).T
tmp = None

# Mueasyre LFP stats for each channel
for ch in range(128):
    
    # Report
    print("Starting channel {0}".format(ch))

    # Extract channel data and convert to uV (float32)
    data_ch = data[ch,:]
    data_ch_uV = (data_ch.astype(np.float32) - 32768) * 0.195
    print("- converted to uV")

# Not sure best LFP stats
# Make example plots...of LFP, compare to raw, etc.

#FIN
