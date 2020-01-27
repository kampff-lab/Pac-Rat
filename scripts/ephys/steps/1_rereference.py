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
    raw_Z[ch,:] = (raw_uV[ch,:] - raw_mean[ch]) / raw_std[ch]

# Specify channels to exclude
#exlcude_channels = np.array([12, 13, 18, 19, 108, 109 ,115])
exlcude_channels = np.array([12, 13, 18, 54, 108, 109 ,115])

# # Plot channel magnitudes
# mags = []
# for i in range(128):
#     mag = np.mean(np.abs(raw_uV[i, :15000]))
#     mags.append(mag)
# plt.figure()
# plt.plot(mags, 'b.')
# plt.show()
# print(np.where(np.array(mags) > 100))

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
A_median = np.median(raw_Z[A_channels,:], axis=0)
B_median = np.median(raw_Z[B_channels,:], axis=0)

# Compute mean values for each headstage
A_mean = np.mean(raw_Z[A_channels,:], axis=0)
B_mean = np.mean(raw_Z[B_channels,:], axis=0)

# Rereference each channel
clean_Z = np.zeros(raw_Z.shape)
for ch in A_channels:
    raw_Z_ch = raw_Z[ch, :]
    clean_Z_ch = raw_Z_ch - A_mean
    clean_Z[ch,:] = clean_Z_ch
for ch in B_channels:
    raw_Z_ch = raw_Z[ch, :]
    clean_Z_ch = raw_Z_ch - B_mean
    clean_Z[ch,:] = clean_Z_ch

# Plot Z-scored ephys data
plt.figure()

# cleaned
probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)
plt.subplot(1,2,1)
offset = 0
colors = cm.get_cmap('tab20b', 11)
for shank in range(11):
    for depth in range(11):
        ch = (depth * 11) + shank
        plt.plot(probe_Z[ch, 142000:155000] + offset, color=colors(shank))
        offset += 2
# raw
plt.subplot(1,2,2)
probe_Z = ephys.apply_probe_map_to_amplifier(raw_Z)
offset = 0
colors = cm.get_cmap('tab20b', 11)
for shank in range(11):
    for depth in range(11):
        ch = (depth * 11) + shank
        plt.plot(probe_Z[ch, 142000:155000] + offset, color=colors(shank))
        offset += 2
plt.show()

# Spectrogram test
plt.figure()
shank = 4
for depth in range(11):
    plt.subplot(11,2,depth*2 + 1)
    probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)
    fs = 30000
    ch = (depth * 11) + shank
    f, t, Sxx = signal.spectrogram(probe_Z[ch,:], fs, nperseg=30000, nfft=30000, noverlap=27000)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim([0, 30])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(11,2,depth*2 + 2)
    plt.plot(probe_Z[ch,:])
plt.show()

#FIN