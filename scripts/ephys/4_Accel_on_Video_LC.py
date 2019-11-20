# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

base_path = r'E:/AK_33.2_test/2018_04_29-15_43'

# Load sampes_for_frames
samples_for_frames_file_path = base_path + r'Analysis/samples_for_frames.csv'
sample_for_each_video_frame = np.genfromtxt(samples_for_frames_file_path, delimiter=',', usecols=0, dtype=np.uint32)

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
video_width = np.int32(video_width)
video_height = np.int32(video_height)

# Loop through video
frame_count = 50000
while(True):
    
    # Make go super fast!
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    
    # Capture frame-by-frame
    ret, frame = video.read()
    
    # Extract ephys around frame
    sample = sample_for_each_video_frame[frame_count]
    window_size = video_width
    window = reshaped_data_T[probe_map_as_vector, sample:(sample+window_size)]
    window = np.int32(window)
    
    # Draw ephys data on image
    for n,ch in enumerate([7, 25, 23, 5, 9, 27, 21, 14, 11, 16, 17]):
        x_pts = np.arange(0,video_width, dtype=np.int32)
        y_pts = (window[ch,:] - 31500 + (n*1000)) // 10
        y_pts = (-1*y_pts)+video_height
        pts = np.vstack((x_pts, y_pts)).T
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],False,(0,255,255))
    
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    ## Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Increment frame counter
    frame_count = frame_count + 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()

# FIN                                
    