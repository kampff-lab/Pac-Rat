# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:12:19 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import os

#base_directory = r'E:'
#rat_ID = r'/AK_33.2_test/'
#rat_folder = base_directory + rat_ID
#day = rat_folder + '2018_04_29-15_43/'
#filename = day + 'impedance1.csv'


#main folder rat ID
#script_dir = os.path.dirname(day)
#create a folder where to store the plots inside the daily sessions
#session=os.path.join(script_dir,'Analysis')
#create a folder where to save the plots
#results_dir = os.path.join(session, 'Daily_Health_Check/')
#plot name
#sample_file_name_heatmap = "Impedance_heatmap"

#if not os.path.isdir(results_dir):
    #os.makedirs(results_dir)

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

def GET_data_zero_mean_remapped_window(filename, offset, num_samples):
    
    num_channels = 128
    bytes_per_sample = 2
    offset_position = offset * num_channels * bytes_per_sample
    
    # Open file and jump to offset position
    f = open(filename, "rb")
    f.seek(offset_position, os.SEEK_SET)

    # Load data from this file position
    data = np.fromfile(f, dtype=np.uint16, count=(num_channels * num_samples))
    f.close()
    
    # Reshape data
    reshaped_data = np.reshape(data,(num_samples,128)).T
    #to have 128 rows
    
    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    data_uV = (reshaped_data.astype(np.float32) - 32768) * 0.195
    
    # Subtract channel mean from each channel
    mean_per_channel_data_uV = np.mean(data_uV,axis=1,keepdims=True)
    data_zero_mean = data_uV - mean_per_channel_data_uV
    
    # Extract (remapped) 121 probe channels
    probe_map_as_vector = np.reshape(probe_map.T, newshape=(121))
    data_zero_mean_remapped = data_zero_mean[probe_map_as_vector,:]
    
    return data_zero_mean_remapped



#def data_highpass(data_zero_mean):
#    highpass_matrix = np.zeros((128,18000000),dtype=float)
#   count=0
#    for i in arange(128):
#        high=highpass(data_zero_mean[i,:],BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
#        highpass_matrix[i,:]=high
#        high[i]=None
#    count += 1
#    return highpass_matrix


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




#main folder rat ID
#script_dir = os.path.dirname(day)
#create a folder where to store the plots inside the daily sessions
#session=os.path.join(script_dir,'Analysis')
#create a folder where to save the plots
#results_dir = os.path.join(session, 'Daily_Health_Check/')
#plot name
#sample_file_name_heatmap = "Impedance_heatmap"

#if not os.path.isdir(results_dir):
#    os.makedirs(results_dir)





   
# Plot corr
# -----------------------------------------------------------------------------

# Create random sample list
outpath = 'E:/AK_33.2_test/2018_04_29-15_43/test_folder/'
window_size = 90000
offset_list=list(range(0,30000 * 60 * 30, 30000))


#random_samples = random.sample(range(0,ten_min_samples), num_corr_samples)
#corr_matrix2 = np.zeros((121,121),dtype=float)
#norm_corr_matrix2 = np.zeros((121,121),dtype = float)
for i in range(len(offset_list)):
    corr_matrix2 = np.zeros((121,121),dtype=float)
    data_zero_mean_remapped = GET_data_zero_mean_remapped_window(filename, offset_list[i], window_size)
    
    for e in range(0, window_size, 30):
        outer_product = np.outer(data_zero_mean_remapped[:, e], data_zero_mean_remapped[:, e])
        corr_matrix2 = corr_matrix2 + outer_product
    corr_matrix2 = corr_matrix2 / window_size / 30   
    
    norm_corr_matrix2 = np.zeros((121,121),dtype = float)
    for r in range(121):
        for c in range(121):
            normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
            norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor
    plt.figure()
    ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right"))
    plt.savefig(outpath +"correlation{filecount}.png".format(filecount=i))
    plt.close('all')
    print("Current offset: " + str(i))










def data_highpass(data_zero_mean):
    highpass_matrix = np.zeros((128,18000000),dtype=float)
    count=0
    for i in arange(128):
        high=highpass(data_zero_mean[i,:],BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
        highpass_matrix[i,:]=high
        high[i]=None
    count += 1
    return highpass_matrix

   
def data_lowpass(data_zero_mean,lowcut=250):
    lowpass_matrix = np.zeros((128,18000000),dtype=float)
    count=0
    for i in arange(128):
        low=butter_filter_lowpass(data_zero_mean[i,:], lowcut,  fs=30000, order=3, btype='lowpass')
        lowpass_matrix[i,:]=low
        low[i]=None
    count += 1
    return lowpass_matrix





highpass_matrix=data_highpass(data_zero_mean)
highpass_matrix_remapped= highpass_matrix[probe_map_as_vector,:]
highpass_matrix=None
highpass_matrix_remapped=None

lowpass_matrix =data_lowpass(data_zero_mean,lowcut=250)







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

