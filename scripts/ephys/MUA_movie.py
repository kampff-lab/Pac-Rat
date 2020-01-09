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
fps = 60
outputVid = cv2.VideoWriter(save_path, fourcc, fps, (video_height, video_width), True)

# Set to start frame
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Draw MUA matrix on video frame
num_frame = 100 # Just for testing
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
#
#
#   csv_path = output_name + '.csv'
#    avi_path = output_name + '.avi'
#    
#    # Open output movie file, then specify compression, frames per second and size
#    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
#    fps = 120
#    outputVid = cv2.VideoWriter(avi_path, fourcc, fps, (window_size, window_size), True)
#    
#    # Compute crop size (half of window size)
#    crop_size = np.int(window_size / 2)
#       
#    # Read current frame
#    success, image = video.read()
#
#    # Measure dimensions
#    width = image.shape[1]
#    height = image.shape[0]
#    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#
#    # Create "larger" frame to contain video frame with border
#    container = np.zeros((height + 2 * crop_size, width + 2 * crop_size, 3), dtype=np.uint8)
#
#    # Create empty frame to contain crop window
#    crop = np.zeros((window_size, window_size,3), dtype=np.uint8)
#
#    # Reset video to first frame
#    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#    
#    # Empty list to be filled with centroid x and y 
#    centroid_x = []
#    centroid_y = []
#    
#    # Read and process each frame
#    cv2.namedWindow("Display")
#    #num_frames = 1200
#    for frame in range(0, num_frames):
#
#        # Read current frame
#        success, image = video.read()
#
#        # Convert to grayscale
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#        # Subtract background (darker values positive)
#        subtracted = cv2.subtract(background, gray)
#
#        # Threshold
#        level, threshed = cv2.threshold(subtracted, 5, 255, cv2.THRESH_BINARY)
#
#        # Open (Erode then Dialate) to remove tail and noise
#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
#        opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
#        
#        # Find Binary Contours            
#        ret, contours, hierarchy = cv2.findContours(np.copy(opened),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)        
#        if len(contours) == 0:
#            # Store NaN in list
#            centroid_x.append(np.nan)
#            centroid_y.append(np.nan)
#            
#            # Store "previous" crop image
#            outputVid.write(crop)            
#            continue
#
#        # Get largest particle
#        largest_cnt, area = get_largest_contour(contours)
#
#        # Get centroid (and moments)          
#        M = cv2.moments(largest_cnt)
#        cx = (M['m10']/M['m00'])
#        cy = (M['m01']/M['m00'])
#                
#        # Insert grayscale frame into the container
#        container[crop_size:(height+crop_size), crop_size:(width+crop_size), :] = image
#        
#        # Offset centroid position to container coordinates
#        container_cx = np.int(cx + crop_size)
#        container_cy = np.int(cy + crop_size)
#        
#        # Crop around the centroid position in the container image
#        crop = container[(container_cy - crop_size):(container_cy + crop_size), (container_cx - crop_size):(container_cx + crop_size),:]
#        
#        # Store centroid in list
#        centroid_x.append(cx)
#        centroid_y.append(cy)
#
#        # Write output video frame
#        outputVid.write(crop)
#                     
#        # Display (occasionally)
##        if (frame % 12) == 0:
##            resized = cv2.resize(gray, (np.int(width/2), np.int(height/2)))
##            color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
##            cv2.circle(color, (np.int(cx/2), np.int(cy/2)), 5, (0, 255, 0), thickness=1, lineType=8, shift=0)
##            cv2.imshow("Display", color)    
##            cv2.waitKey(1)
##            print("Frame %d of %d" % (frame, num_frames))
#
#    # Save csv file containing the centroids coordinates    
#    np.savetxt(csv_path, np.vstack((centroid_x,centroid_y)).T,delimiter=',')    
#
#    # Cleanup
#    cv2.destroyAllWindows()
#    
#    # Clouse output video
#    outputVid.release() 
#    return 1
#




























