# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:59:46 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import os



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
filename = 'F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/Amplifier.bin'

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


#extract tone sample from sync file

sync_file= 'F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/Sync.bin'
sync_dtype = np.uint8
fs = 30000
sync_data = np.fromfile(sync_file,sync_dtype)
tone_sync_data = np.int8(sync_data & 8)
tone_transitions = np.diff(tone_sync_data)
tone_starts = np.where(tone_transitions == 8)[0]
tone_starts=tone_starts.tolist()

# Remove missed trials (manually)
del tone_starts[2]
num_tones = len(tone_starts)
list_availability_tone = []
list_reward_tone = []

# Make list of tone types 
for i in range(0, num_tones, 2):
    list_availability_tone.append(tone_starts[i])
    list_reward_tone.append(tone_starts[i + 1])
 
    
# Select tone type to plot
list_tones = list_reward_tone
#list_tones = list_availability_tone
channels_to_avg = range(0,55,1)

# known tone
pre_offset = 10000
post_offset = 100000
plt.figure()
all_data_zero_mapped = np.zeros((121, pre_offset + post_offset, len(list_tones)))
count = 0
for offset in list_tones:
    all_data_zero_mapped[:,:, count] = GET_data_zero_mean_remapped_window(filename, offset - pre_offset, post_offset + pre_offset)
    count += 1

# Plot avg data
avg_data_zero_mapped = np.mean(all_data_zero_mapped, 2)
plt.figure()
for i in range(0, 121):
    plt.plot(avg_data_zero_mapped[i, :] + (20 * i), color=[0.0,0.0,0.0,0.5], LineWidth=1.0)

# Plot full average
plt.figure()
full_avg = np.mean(avg_data_zero_mapped[channels_to_avg,:], 0)
plt.plot(full_avg, color=[1.0,0.0,0.0,1.0], LineWidth=3.0)




















base_path = r'E:/AK_33.2_test/2018_04_29-15_43/'

# Load sampes_for_frames
samples_for_frames_file_path = base_path + r'Analysis_old/samples_for_frames.csv'
sample_for_each_video_frame = np.genfromtxt(samples_for_frames_file_path, delimiter=',', usecols=0, dtype=np.uint32)



def closest_value_in_array(array,value_list):
    nearest  = []
    for e in value_list:
        delta = array-e
        nearest.append(np.argmin(np.abs(delta)))
    return nearest   


video_avi_file_path='E:/AK_33.2_test/2018_04_29-15_43/Video.avi'
target_dir= r'E:\AK_33.2_test\2018_04_29-15_43\tone_frames'

def tone_frame(target_dir,video_avi_file_path,nearest):
    video=cv2.VideoCapture(video_avi_file_path)
    success, image=video.read()
    success=True
    count = 0
    for i in nearest:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = video.read()
        if count < 10:
            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
        else:
            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
        count += 1
    return image



nearest_availability_tone=closest_value_in_array(sample_for_each_video_frame,list_availability_tone)


nearest_reward_tone=closest_value_in_array(sample_for_each_video_frame,list_reward_tone)


