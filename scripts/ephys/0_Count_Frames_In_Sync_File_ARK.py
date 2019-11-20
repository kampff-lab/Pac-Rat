# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt

base_path = r'G:/AK_33.2/2018_05_14-15_09/'


# Load Video File Details
video_csv_file_path = base_path + 'video.csv'
video_counter = np.genfromtxt(video_csv_file_path, delimiter=' ', usecols=1)
video_num_frames = video_counter[-1] - video_counter[0] + 1

# Load Sync File Details                       
sync_file_path = base_path + 'sync.bin'
sync_dtype = np.uint8
fs = 30000
frame_rate = 120.0
sync_data = np.fromfile(sync_file_path,sync_dtype)
frame_sync_data = np.int8(sync_data & 1)
frame_transitions = np.diff(frame_sync_data)
frame_starts = np.where(frame_transitions == 1)[0]
sync_num_frames = len(frame_starts)
if video_num_frames != sync_num_frames:
    print("DISASTER")
else:
    print("Not Disaster")
                
# FIN                                
    
#t = sync_data & 4 > 0