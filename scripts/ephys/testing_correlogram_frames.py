# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:04:58 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
import seaborn as sns
from matplotlib import gridspec

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


idx_closest_offset_to_video_sample = closest_value_in_array(sample_for_each_video_frame,offset_list)
offset_video_sample = sample_for_each_video_frame[idx_closest_offset_to_video_sample]



'E:/AK_33.2_test/2018_04_29-15_43/Analysis_old/samples_for_frames.csv'
# Load Ephys data

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
probe_map_as_vector = np.reshape(probe_map.T, newshape=(121))

# Load Data as uint16 from binary file, use memory mapping (i.e. do not load into RAM)
#   - use read-only mode "r+" to prevent overwriting the original file
ephys_file_path = base_path + r'Amplifier.bin'
num_channels = 128
data = np.memmap(ephys_file_path, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
reshaped_data_T= reshaped_data.T
data = None
reshaped_data = None

# Load Video CSV
video_csv_file_path = base_path + 'Video.csv'
video_counter = np.genfromtxt(video_csv_file_path, delimiter=' ', usecols=1)

# Load Video AVI
video_avi_file_path = base_path + 'Video.avi'
video = cv2.VideoCapture(video_avi_file_path)
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)



video_avi_file_path='E:/AK_33.2_test/2018_04_29-15_43/Video.avi'

target_dir= r'E:\AK_33.2_test\2018_04_29-15_43\test_frames'

def frame_before_trials(target_dir,video_avi_file_path,offset_list):
    video=cv2.VideoCapture(video_avi_file_path)
    success, image=video.read()
    success=True
    count = 0
    for i in offset_list[:10]:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = video.read()
        if count < 10:
            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
        else:
            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
        count += 1
    return image


frame_before_trials(target_dir,video_avi_file_path,offset_list)


# Display the resulting frame
#cv2.imshow('frame',frame)
    





# Create random sample list
outpath = 'E:/AK_33.2_test/2018_04_29-15_43/test_folder/'
window_size = 90000
offset_list=list(range(0,30000 * 60 * 30, 30000))


#random_samples = random.sample(range(0,ten_min_samples), num_corr_samples)
#corr_matrix2 = np.zeros((121,121),dtype=float)
#norm_corr_matrix2 = np.zeros((121,121),dtype = float)


video=cv2.VideoCapture(video_avi_file_path)

success=True

for i in range(len(offset_list)):
    corr_matrix2 = np.zeros((121,121),dtype=float)
    data_zero_mean_remapped = GET_data_zero_mean_remapped_window(filename, offset_list[i], window_size)
    video.set(cv2.CAP_PROP_POS_FRAMES, sample_for_each_video_frame[i])
    success, image = video.read()
    success=True


    for e in range(0, window_size, 30):
        outer_product = np.outer(data_zero_mean_remapped[:, e], data_zero_mean_remapped[:, e])
        corr_matrix2 = corr_matrix2 + outer_product
    corr_matrix2 = corr_matrix2 / window_size / 30   
    
    norm_corr_matrix2 = np.zeros((121,121),dtype = float)
    for r in range(121):
        for c in range(121):
            normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
            norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor 
  
    fig = plt.figure(figsize=(5, 8))
    a = fig.add_subplot(2, 1, 1)
    imgplot = plt.imshow(image)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax = fig.add_subplot(2, 1, 2)
    #fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right"),cmap="YlGnBu",vmax=1, vmin=-0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    #plt.subplots_adjust(hspace = 0)
    plt.savefig(target_dir +"\correlation{filecount}.png".format(filecount=i))
    plt.close('all')
    print("Current offset: " + str(i))












gs1 = gridspec.GridSpec(2, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])




fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 3]) 
ax0 = plt.subplot(gs[0])
ax0.plot(x, y)
ax1 = plt.subplot(gs[1])
ax1.plot(y, x)

plt.tight_layout()      











