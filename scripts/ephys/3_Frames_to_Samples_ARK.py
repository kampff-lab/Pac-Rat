# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt

base_path = r'F:/AK_33.2/2018_04_29-15_43/'

# Load Video File Details
video_csv_file_path = base_path + 'video.csv'
video_counter = np.genfromtxt(video_csv_file_path, delimiter=' ', usecols=1)

# Load Sync File Details                       
sync_file_path = base_path + 'sync.bin'
sync_dtype = np.uint8
fs = 30000
frame_rate = 120.0
sync_data = np.fromfile(sync_file_path,sync_dtype)
frame_sync_data = np.int8(sync_data & 1)
frame_transitions = np.diff(frame_sync_data)
frame_starts = np.where(frame_transitions == 1)[0]
                
# Get sample number for each video frame
num_video_frames = len(video_counter)
sync_indices_for_video_frames = (video_counter-video_counter[0]).astype(np.uint32)
sample_for_each_video_frame = frame_starts[sync_indices_for_video_frames]
np.savetxt(base_path + r'Analysis/samples_for_frames.csv', sample_for_each_video_frame, delimiter=',')

# FIN                                
    