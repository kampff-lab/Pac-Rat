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
from scipy import stats
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

# Determine linear scaling model for each channel
clean = np.zeros(raw.shape)
for ch in A_channels:
    raw_ch = raw[ch, :]
    mean_ch = A_mean
    median_ch = A_median
    sorted_ch = np.sort(raw_ch)
    lower_thresh = sorted_ch[np.int(num_samples * 0.01)]
    upper_thesh = sorted_ch[np.int(num_samples * 0.99)]
    valid = (raw_ch < lower_thresh) + (raw_ch > upper_thesh)
    m, b, r_value, p_value, std_err = stats.linregress(raw_ch[valid], mean_ch[valid])
    scaled_ch = m*raw_ch + b
    clean_ch = scaled_ch - mean_ch
    clean[ch,:] = clean_ch
for ch in B_channels:
    raw_ch = raw[ch, :]
    mean_ch = B_mean
    median_ch = B_median
    sorted_ch = np.sort(raw_ch)
    lower_thresh = sorted_ch[np.int(num_samples * 0.01)]
    upper_thesh = sorted_ch[np.int(num_samples * 0.99)]
    valid = (raw_ch < lower_thresh) + (raw_ch > upper_thesh)
    m, b, r_value, p_value, std_err = stats.linregress(raw_ch[valid], mean_ch[valid])
    scaled_ch = m*raw_ch + b
    clean_ch = scaled_ch - mean_ch
    clean[ch,:] = clean_ch

# Report cleaning
ch = 21
raw_ch = raw[ch, :]
mean_ch = A_mean
median_ch = A_median
sorted_ch = np.sort(raw_ch)
lower_thresh = sorted_ch[np.int(num_samples * 0.01)]
upper_thesh = sorted_ch[np.int(num_samples * 0.99)]
valid = (raw_ch < lower_thresh) + (raw_ch > upper_thesh)
m, b, r_value, p_value, std_err = stats.linregress(raw_ch[valid], mean_ch[valid])
scaled_ch = m*raw_ch + b
clean_ch = scaled_ch - mean_ch

plt.figure()
plt.plot(raw_ch[valid], mean_ch[valid], 'k.')
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(raw_ch, 'r')
plt.plot(mean_ch, 'g')
plt.plot(scaled_ch, 'b')
plt.subplot(2,1,2)
plt.plot(clean_ch, 'k')
plt.show()

# Plot cleaned vs. raw ephys data
plt.figure()

# cleaned
probe = ephys.apply_probe_map_to_amplifier(clean)
plt.subplot(1,2,1)
offset = 0
colors = cm.get_cmap('tab20b', 11)
for shank in range(11):
    for depth in range(11):
        ch = (depth * 11) + shank
        plt.plot(probe[ch, 142000:155000] + offset, color=colors(shank))
        offset += 100
# raw
plt.subplot(1,2,2)
probe = ephys.apply_probe_map_to_amplifier(raw)
offset = 0
colors = cm.get_cmap('tab20b', 11)
for shank in range(11):
    for depth in range(11):
        ch = (depth * 11) + shank
        plt.plot(probe[ch, 142000:155000] + offset, color=colors(shank))
        offset += 100
plt.show()

#FIN
