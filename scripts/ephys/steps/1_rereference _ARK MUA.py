# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: median/mean rereferencing

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
raw_path = os.path.join(session_path +'/Amplifier.bin')

# Specify sample range for clip
start_sample = 32837802
num_samples = 657331
num_samples = 1657331

# Specify channel of interest
depth = 6
shank = 10

# Load raw data and convert to microvolts
raw_uV = ephys.get_raw_clip_from_amplifier(raw_path, start_sample, num_samples)

# Compute mean and standard deviation for each channel
raw_mean = np.mean(raw_uV, axis=1)
raw_std = np.std(raw_uV, axis=1)

# Z-score each channel
raw_Z = np.zeros(raw_uV.shape)
for ch in range(128):
    raw_Z[ch,:] = (raw_uV[ch,:] - raw_mean[ch])# / raw_std[ch]

# Store raw Z-scored as raw
raw = np.copy(raw_Z)

# Specify channels to exclude
#exlcude_channels = np.array([12, 13, 18, 19, 108, 109 ,115])
exlcude_channels = np.array([12, 13, 18, 54, 108, 109 ,115])

# Determine channels to exclude on each headstage
A_exclude_channels = exlcude_channels[exlcude_channels < 64]
B_exclude_channels = exlcude_channels[exlcude_channels >= 64]

# Determine headstage channels
A_channels = np.arange(64)
B_channels = np.arange(64, 128)

# Remove excluded channels
A_channels = np.delete(A_channels, A_exclude_channels)
B_channels = np.delete(B_channels, B_exclude_channels)

# Compute median values for each headstage
A_median = np.median(raw[A_channels,:], axis=0)
B_median = np.median(raw[B_channels,:], axis=0)

# Compute mean values for each headstage
A_mean = np.mean(raw[A_channels,:], axis=0)
B_mean = np.mean(raw[B_channels,:], axis=0)

# Rereference each channel
clean = np.zeros(raw.shape)
for ch in A_channels:
    raw_ch = raw[ch, :]
    clean_ch = raw_ch - A_mean
    clean[ch,:] = clean_ch
for ch in B_channels:
    raw_ch = raw[ch, :]
    clean_ch = raw_ch - B_mean
    clean[ch,:] = clean_ch

## Plot Z-scored ephys data
#plt.figure()
#
## cleaned
#probe = ephys.apply_probe_map_to_amplifier(clean)
#plt.subplot(1,2,1)
#offset = 0
#colors = cm.get_cmap('tab20b', 11)
#for shank in range(11):
#    for depth in range(11):
#        ch = (depth * 11) + shank
#        plt.plot(probe[ch, 142000:155000] + offset, color=colors(shank))
#        offset += 2
## raw
#plt.subplot(1,2,2)
#probe = ephys.apply_probe_map_to_amplifier(raw)
#offset = 0
#colors = cm.get_cmap('tab20b', 11)
#for shank in range(11):
#    for depth in range(11):
#        ch = (depth * 11) + shank
#        plt.plot(probe[ch, 142000:155000] + offset, color=colors(shank))
#        offset += 2
#plt.show()

# Measure threshold crossings
signal = ephys.apply_probe_map_to_amplifier(clean)
num_channels = len(signal)
spike_times = [[] for _ in range(num_channels)]  
spike_peaks = [[] for _ in range(num_channels)]  

for channel in np.arange(num_channels):

    try:
        # Extract data for single channel
        channel_data = signal[channel,:]
        
        # FILTERS (one ch at the time)
        channel_data_highpass = ephys.highpass(channel_data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    
        # Determine high and low threshold
        abs_channel_data_highpass = np.abs(channel_data_highpass)
        sigma_n = np.median(abs_channel_data_highpass) / 0.6745
        
        #adaptive th depending of ch noise
        spike_threshold_hard = -3.0 * sigma_n
        spike_threshold_soft = -1.0 * sigma_n
        
        # Find threshold crossings
        spike_start_times, spike_stop_times = ephys.threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft)    
        
        # Find peak voltages and times
        spike_peak_voltages = []
        spike_peak_times = []
        for start, stop in zip(spike_start_times,spike_stop_times):
            peak_voltage = np.min(channel_data_highpass[start:stop]) 
            peak_voltage_idx = np.argmin(channel_data_highpass[start:stop])
            spike_peak_voltages.append(peak_voltage)
            spike_peak_times.append(start + peak_voltage_idx)
        
        # Remove too early and too late spikes
        spike_starts = np.array(spike_start_times)
        spike_stops = np.array(spike_stop_times)
        peak_times = np.array(spike_peak_times)
        peak_voltages = np.array(spike_peak_voltages)
        good_spikes = (spike_starts > 100) * (spike_starts < (len(channel_data_highpass)-200))
    
        # Select only good spikes
        spike_starts = spike_starts[good_spikes]
        spike_stops = spike_stops[good_spikes]
        peak_times = peak_times[good_spikes]
        peak_voltages = peak_voltages[good_spikes]
        
        #peak_times_corrected  = start_sample + peak_times
        #spike_times_Z[channel] = peak_times_corrected
        #spike_times_clean_model[channel] = peak_times_corrected
        #spike_times_raw[channel] = peak_times_corrected
        #spike_times_shank[channel] = peak_times_corrected
        #spike_times_no_Z[channel] = peak_times_corrected
        
        spike_times[channel] = peak_times
        spike_peaks[channel] = peak_voltages
        print(channel)
        
    except Exception:
        continue

#FIN