# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 2a: health check analysis and plots for MUA signal (high-frequency)

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

# Specify cleaned data path
data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')

# Measure file size (and number of samples)
bytes_per_sample = 16
statinfo = os.stat(data_path)
num_samples = np.int(statinfo.st_size / bytes_per_sample)
num_samples_per_channel = np.int(num_samples / num_raw_channels)

# Memory map amplifier data
tmp = np.memmap(data_path, dtype=np.uint16, mode = 'r')
data = np.reshape(tmp,(num_samples_per_channel,128)).T
tmp = None

# Measure stats for each channel
for ch in range(128):
    
    # Report
    print("Starting channel {0}".format(ch))

    # Extract channel data and convert to uV (float32)
    data_ch = data[ch,:]
    data_ch_uV = (data_ch.astype(np.float32) - 32768) * 0.195
    print("- converted to uV")

    # High-pass filter at 500 Hz
    highpass_ch_uV = ephys.highpass(data_ch_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    print("- highpassed")

    # Determine RMS noise after filtering
    abs_highpass_ch_uV = np.abs(highpass_ch_uV)
    sigma_n = np.median(abs_highpass_ch_uV) / 0.6745
    print("- Noise sigma measured")

# Measure RMS of each channel (in uV) and save
# Make example plots...of unfiltered and high-pass filtered data
# Should compare this with both "cleaned" and raw data

#FIN
