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
filename = 'E:/AK_33.2_test/2018_04_29-15_43/Amplifier.bin'
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

# Extract data chunk
minutes = 10
seconds = minutes*60
ten_min_samples = seconds*freq
centre = int(num_samples/2)
interval = int(ten_min_samples/2)
data_ten_min_chunk = reshaped_data_T[:,centre-interval:centre+interval]
reshaped_data_T = None



# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
data_uV = (data_ten_min_chunk.astype(np.float32) - 32768) * 0.195
data_ten_min_chunk = None

# Subtract channel mean from each channel
mean_per_channel_data_uV = np.mean(data_uV,axis=1,keepdims=True)
data_zero_mean = data_uV - mean_per_channel_data_uV

# Extract (remapped) 121 probe channels
probe_map_as_vector = np.reshape(probe_map.T, newshape=(121))
data_zero_mean_remapped = data_zero_mean[probe_map_as_vector,:]




def data_highpass(data_zero_mean):
    highpass_matrix = np.zeros((128,18000000),dtype=float)
    count=0
    for i in arange(128):
        high=highpass(data_zero_mean[i,:],BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
        highpass_matrix[i,:]=high
        high[i]=None
    count += 1
    return highpass_matrix

#
#    
#def data_lowpass(data_zero_mean,lowcut=250):
#    lowpass_matrix = np.zeros((128,18000000),dtype=float)
#    count=0
#    for i in arange(128):
#        low=butter_filter_lowpass(data_zero_mean[i,:], lowcut,  fs=30000, order=3, btype='lowpass')
#        lowpass_matrix[i,:]=low
#        low[i]=None
#    count += 1
#    return lowpass_matrix
#
#



highpass_matrix=data_highpass(data_zero_mean)
highpass_matrix_remapped= highpass_matrix[probe_map_as_vector,:]


























   
# Plot corr
# -----------------------------------------------------------------------------

# Create random sample list
num_corr_samples = 10000
random_samples = random.sample(range(0,ten_min_samples), num_corr_samples)
corr_matrix = np.zeros((121,121),dtype=float)
for i in range(num_corr_samples):
        outer_product = np.outer(data_zero_mean_remapped[:,random_samples[i]], data_zero_mean_remapped[:,random_samples[i]])
        corr_matrix = corr_matrix + outer_product
corr_matrix = corr_matrix / num_corr_samples

# Normalize by the value of each channels auto-correlation
norm_corr_matrix = np.zeros((121,121),dtype = float)
for r in range(121):
    for c in range(121):
        normalization_factor = (corr_matrix[r,r] + corr_matrix[c,c])/2
        norm_corr_matrix[r,c] = corr_matrix[r,c]/normalization_factor

sample_file_name_corr_matrix = "Correaltion_matrix"
plt.figure()
#ax = sns.heatmap(zero_mean_map,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
ax = sns.heatmap(norm_corr_matrix, cbar_kws = dict(use_gridspec = False,location = "right"))
#plt.xlabel("Colors")
#plt.ylabel("Values")
plt.title("Normalized Correlation Matrix")
plt.show()
plt.savefig(results_dir + sample_file_name_corr_matrix)



#PLot mean
#------------------------------------------------------------------------------


mean_per_channel_data_uV_remapped = mean_per_channel_data_uV[probe_map_as_vector,:]

plt.figure()
sample_file_name_data_mean = "Data_mean"
plt.plot(mean_per_channel_data_uV_remapped)
sns.set_style("white")
plt.xlabel("channels_mapped")
plt.ylabel("mean")
plt.title("data_mean")
plt.savefig(results_dir + sample_file_name_data_mean,facecolor=fig.get_facecolor(), edgecolor='none')



#Plot zero mean
#------------------------------------------------------------------------------

mean_data_zero_mean_remapped = np.mean(data_zero_mean_remapped, axis=1)

mean_data_zero_mean_remapped_heatmap = np.reshape(mean_data_zero_mean_remapped,newshape=probe_map.shape)


plt.figure()
sample_file_name_data_zero_mean_heatmap = "Data_zero_mean_heatmap"
#ax = sns.heatmap(zero_mean_map,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
ax = sns.heatmap(mean_data_zero_mean_remapped_heatmap,annot = True,annot_kws = {"size": 7},cbar_kws = dict(use_gridspec = False,location = "right"))
#plt.xlabel("Colors")
#plt.ylabel("Values")
plt.title("zero mean")
plt.savefig(results_dir + sample_file_name_data_zero_mean_heatmap)


plt.figure()
sample_file_name_data_zero_mean = "Data_zero_mean"
plt.plot(mean_data_zero_mean_remapped)
plt.xlabel("channels_mapped")
plt.ylabel("mean_zero_mean")
plt.title("zero mean")
plt.savefig(results_dir + sample_file_name_data_zero_mean,facecolor=fig.get_facecolor(), edgecolor='none')




#FILTERS (one ch at the time)

#channel_number = 55
#data_highpass = highpass(data_zero_mean[channel_number,:],BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)

#plt.plot(data_highpass[100000:105000])

#data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')

#plt.plot(data_lowpass[100000:110000])
#plt.plot(data_zero_mean[55][100000:105000])

#CALCULATE STD (on zero mean data) after filtering (for loop which filter one ch at the time)
#this is required to see how much the signal varies on every channel 

#estimate of the noise
h_std=std_highpass(data_zero_mean)

#estimate of the quality of the LFP
l_std=std_lowpass(data_zero_mean,lowcut=250)
   




h_std_mapped = h_std[probe_map_as_vector]
data_h_std_mapped = np.reshape(h_std_mapped,newshape=probe_map.shape)



plt.figure()
sample_file_name_data_std_h_mean_heatmap = "center_10'_highpass_std"
ax = sns.heatmap(data_h_std_mapped,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
#plt.xlabel("Colors")
#plt.ylabel("Values")
plt.title("Highpass std")
plt.savefig(results_dir + sample_file_name_data_std_h_mean_heatmap)

# MAP LOW pass data + HEATMAP

l_std_mapped = l_std[probe_map_as_vector]
data_l_std_mapped =np.reshape(l_std_mapped,newshape=probe_map.shape)



plt.figure()
sample_file_name_data_std_l_mean_heatmap = "center_10'_lowpass_std"
ax = sns.heatmap(data_l_std_mapped ,annot=True,annot_kws={"size": 7}, cbar_kws = dict(use_gridspec=False,location="right"))
#plt.xlabel("Colors")
#plt.ylabel("Values")
plt.title("Lowpass std")
plt.savefig(results_dir + sample_file_name_data_std_l_mean_heatmap)































# FIN                                
    