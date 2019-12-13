# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:44 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from filters import *

### Load and pre-process data

# Probe from superficial to deep electrode, left side is shank 11 (far back)
probe_map=np.array([[103,78,81,118,94,74,62,24,49,46,7],
                    [121,80,79,102,64,52,32,8,47,48,25],
                    [123,83,71,104,66,84,38,6,26,59,23],
                    [105,69,100,120,88,42,60,22,57,45,5],
                    [101,76,89,127,92,67,56,29,4,37,9],
                    [119,91,122,99,70,61,34,1,39,50,27],
                    [112,82,73,97,68,93,40,3,28,51,21],
                    [107,77,98,125,86,35,58,31,55,44,14],
                    [110,113,87,126,90,65,54,20,2,43,11],
                    [117,85,124,106,72,63,36,0,41,15,16],
                    [114,111,75,96,116,95,33,10,30,53,17]])



#   - use read-only mode "r+" to prevent overwriting the original file
recording = 'F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/Amplifier.bin'
samples_for_frames_file_path ='F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/Analysis/samples_for_frames.csv'
samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype=int)

num_channels = 128
freq = 30000

flatten_probe = probe_map.flatten()
lowcut = 250
highcut = 2000



for ch,channel in enumerate(flatten_probe):
    
    data = np.memmap(recording, dtype = np.uint16, mode = 'r')
    num_samples = int(int(len(data))/num_channels)
    recording_time_sec = num_samples/freq
    recording_time_min = recording_time_sec/60
    reshaped_data = np.reshape(data,(num_samples,128))
    #to have 128 rows
    reshaped_data_T= reshaped_data.T
    data = None
    reshaped_data = None

    # Extract data chunk for single channel
    channel_data = reshaped_data_T[channel,:]
    reshaped_data_T = None

    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
    channel_data = None
    
    # FILTERS (one ch at the time)
    #channel_data_highpass = highpass(channel_data_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    #data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')
    channel_data_MUA_bandpass = butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order = 3, btype = 'bandpass')
    
    #lowcut = 500
    #highcut = 2000
    #channel_data_bandpass =  butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order=3, btype='bandstop')

    # RASTER CODE
    
    binned_signal = np.zeros((121,len(samples_for_frames)))
    
    
    # Determine high and low threshold
    abs_channel_data_MUA = np.abs(channel_data_MUA_bandpass)
    
    
    sample_diff = np.diff(samples_for_frames)
    sample_diff = np.hstack((sample_diff,250))

    for s in np.arange(len(samples_for_frames)):
        sample = samples_for_frames[s]
        signal_to_bin = abs_channel_data_MUA[sample:(sample + sample_diff[s])]
        avg = np.mean(signal_to_bin)
        binned_signal[ch][s] = avg
        
    print(ch)



















