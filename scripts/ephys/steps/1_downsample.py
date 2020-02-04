# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: downsample to 1 kHz

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import parser_library as parser
import behaviour_library as behaviour
import ephys_library as ephys 

# Reload modules
import importlib
importlib.reload(parser)
importlib.reload(behaviour)
importlib.reload(ephys)

# Specify session folder
session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
#session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

# Specify data paths
raw_path = os.path.join(session_path +'/Amplifier.bin')

# Specify sample range for clip
start_sample = 32837802
num_samples = 657331

# Load raw data and convert to microvolts
raw = ephys.get_raw_clip_from_amplifier(raw_path, start_sample, num_samples)

# Specify channels to exclude
#exlcude_channels = np.array([12, 13, 18, 19, 108, 109 ,115])
exlcude_channels = np.array([12, 13, 18, 54, 108, 109 ,115])

# Downsample each channel
num_ds_samples = np.int(np.floor(num_samples / 30))
downsampled = np.zeros((128, num_ds_samples))
for ch in range(128):
    raw_ch = raw[ch,:]
    lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
    downsampled_ch = lowpass_ch[::30]
    downsampled[ch, :] = downsampled_ch[:num_ds_samples]

# Store downsampled data in a binary file





# Report
ch = 21
raw_ch = raw[ch,:]
lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
downsampled_ch = downsampled[ch, :]
plt.figure()
plt.plot(raw_ch, 'r')
plt.plot(lowpass_ch, 'g')
plt.plot(np.arange(num_ds_samples) * 30, downsampled_ch, 'b')
plt.show()

# LORY (spectral analysis, LFP, etc.)

#FIN