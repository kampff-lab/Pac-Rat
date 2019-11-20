# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from filters import highpass

# Load and pre-process Data
# -----------------------------------------------------------------------------

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

# Load Data as uint16 from binary file, use memory mapping (i.e. do not load into RAM)
#   - use read-only mode "r+" to prevent overwriting the original file
filename = 'F:/AK_31.1/2018_04_03-14_18/Amplifier.bin'
num_channels = 128
data = np.memmap(filename, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
reshaped_data_T= reshaped_data.T
data = None
reshaped_data = None





# Select one channel
channel = 2
channel_data = reshaped_data_T[channel, :]
channel_data_float = channel_data.astype(np.float32)

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
channel_data_uV = (channel_data_float - 32768) * 0.195

# FILTERS (one ch at the time)
channel_data_highpass = highpass(channel_data_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
#data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')

plt.plot(channel_data_highpass[1000000:2100000])
#plt.plot(data_zero_mean[55][100000:105000])

# Find spikes
spike_times = []
spike_threshold_high = -185
spike_threshold_low = -65
spiking = False

if(spike_threshold_high > 0):
    for i, v in enumerate(channel_data_highpass):
        if(not spiking):
            if(v > spike_threshold_high):
                spiking = True
                spike_times.append(i)
        else:
            if(v < spike_threshold_low):
                spiking = False
else:
   for i, v in enumerate(channel_data_highpass):
        if(not spiking):
            if(v < spike_threshold_high):
                spiking = True
                spike_times.append(i)
        else:
            if(v > spike_threshold_low):
                spiking = False       

# Remove too early and too late spikes
spike_times = np.array(spike_times)
spike_times = spike_times[(spike_times > 100) * (spike_times < (len(channel_data_highpass)-200))]

# Plot all spikes
spikes = np.zeros((len(spike_times), 300))
for i, s in enumerate(spike_times):
    spikes[i,:] = channel_data_highpass[(s-100):(s+200)]
plt.figure()
plt.plot(spikes[range(0,len(spike_times), 20),:].T, '-', Color=[0,0,0,.002])
avg_spike = np.mean(spikes, axis=0)
plt.plot(avg_spike, '-', Color=[1,0,0,.5])




# FIN                                
    