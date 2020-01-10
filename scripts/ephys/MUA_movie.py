# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:44 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
#import seaborn as sns
#from filters import *
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs
import behaviour_library as behaviour

# Specify paths
#video_path = '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43/Video.avi'
#mua_path = '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43/MUA_250_to_2000.bin'


rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post

# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)
mua_path = os.path.join(session_path +'/MUA_250_to_2000.bin')

save_path = os.path.join(session_path +'/Overlay.avi')
video_path =  os.path.join(session_path + '/Video.avi')



# Load MUA (binned to frames)
mua_flat_f32 = np.fromfile(mua_path, dtype=np.float32)
mua_channels = np.reshape(mua_flat_f32, (121,-1))
mua = np.reshape(mua_channels, (11,11,-1))

# Compute full movie median (as baseline)
mua_median = np.median(mua, 2)

# Compute full movie stdev (to z-score)
mua_std = np.std(mua, 2)

# Subtract median (zero baseline) and divide by std (z-score)
mua_zeroed = np.zeros(np.shape(mua))
mua_z_score = np.zeros(np.shape(mua))
for r in range(11):
    for c in range(11):
        mua_zeroed[r,c,:] = (mua[r,c,:] - mua_median[r,c])
        mua_z_score[r,c,:] = (mua[r,c,:] - mua_median[r,c]) / mua_std[r,c]

# Measure MUA stats
mua_min = np.min(mua_zeroed[:])
mua_max = np.max(mua_zeroed[:])

# Load Video AVI
video = cv2.VideoCapture(video_path)
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
video_width = np.int32(video_width)
video_height = np.int32(video_height)
num_frames = np.int32(num_frames)

# Create output AVI
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
fps = 120
outputVid = cv2.VideoWriter(save_path, fourcc, fps, (video_height, video_width))

# Set to start frame
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Draw MUA matrix on video frame
#num_frame = 100 # Just for testing
for i in range(num_frames):

    # Capture frame-by-frame
    ret, frame = video.read()

    # rescale MUA
    this_mua = mua_zeroed[:,:,i]
    this_mua[this_mua < 0] = 0
    this_mua[this_mua > 10] = 10
    this_mua = this_mua * 25.5
    this_mua = np.uint8(this_mua)
    
    # Draw MUA data on image
    this_mua_big = cv2.resize(this_mua, (256, 256))
    frame[100:356, 100:356, 0] = this_mua_big
    
    # Display the resulting frame
    #cv2.imshow('frame',frame)
    outputVid.write(frame)
    
# When everything done, release the capture and save video
video.release()
outputVid.release()

# Close windows
cv2.destroyAllWindows()

# FIN






























