# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

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
filename = 'F:/AK_31.1/2018_03_26-15_09/Amplifier.bin'
num_channels=128
data = np.memmap(filename, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
reshaped_data_T= reshaped_data.T
data=None
reshaped_data=None

# Extract data chunk
minutes=10
seconds=minutes*60
ten_min_samples=seconds*freq
centre=int(num_samples/2)
interval=int(ten_min_samples/2)
data_ten_min_chunk=reshaped_data_T[:,centre-interval:centre+interval]
reshaped_data_T=None

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
data_uV = (data_ten_min_chunk.astype(np.float32) - 32768) * 0.195
data_ten_min_chunk=None

# Subtract channel mean from each channel
mean_per_channel_data_uV = np.mean(data_uV,axis=1,keepdims=True)
data_zero_mean = data_uV - mean_per_channel_data_uV

# Extract (remapped) 121 probe channels
probe_map_as_vector = np.reshape(probe_map.T, newshape=(121))
data_zero_mean_remapped = data_zero_mean[probe_map_as_vector,:]






    
# Plot corr
# -----------------------------------------------------------------------------

# Create random sample list
num_corr_samples = 10000
random_samples = random.sample(range(0,ten_min_samples), num_corr_samples)
corr_matrix=np.zeros((121,121),dtype=float)
for i in range(num_corr_samples):
        outer_product = np.outer(data_zero_mean_remapped[:,random_samples[i]], data_zero_mean_remapped[:,random_samples[i]])
        corr_matrix = corr_matrix + outer_product
corr_matrix = corr_matrix / num_corr_samples

# Normalize by the value of each channels auto-correlation
norm_corr_matrix = np.zeros((121,121),dtype=float)
for r in range(121):
    for c in range(121):
        normalization_factor = (corr_matrix[r,r] + corr_matrix[c,c])/2
        norm_corr_matrix[r,c] = corr_matrix[r,c]/normalization_factor


plt.figure()
#ax = sns.heatmap(zero_mean_map,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
ax = sns.heatmap(norm_corr_matrix, cbar_kws = dict(use_gridspec=False,location="right"))
#plt.xlabel("Colors")
#plt.ylabel("Values")
plt.title("Normalized Correlation Matrix")




# FIN                                
    