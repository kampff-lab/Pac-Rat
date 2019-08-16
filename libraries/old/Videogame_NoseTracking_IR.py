# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 10:55:00 2015

@author: adam and lorenza
"""

# Import Required Python Librariers
import numpy as np
import pandas as pd
import cv2
import os 
import matplotlib.pyplot as plt

# Import new librariers
import skimage 
from skimage import feature
from skimage import io
from skimage import morphology


# Define Helper Functions
# ----------------------------------------------------------------------------------

# Load a CSV file containing just timestamps (ISO-8601) and fill a Pandas Dataframe
def timestamp_CSV_to_pandas(filename):
    timestamps = pd.read_csv(filename, header = None, parse_dates=[0])
    return timestamps

# Find the timestamps in one list that are closest in time to those in another (events)
def closest_timestamps_to_events(timestamp_list, event_list):
    nearest  = []
    for e in event_list[0]:
        delta_times = timestamp_list[0]-e
        nearest.append(np.argmin(np.abs(delta_times)))
    return nearest

# Extract a certain number of frames around (before and after) a given event time
def clip_aligned_on_event_to_stack(movie_filename, event_frame, num_before, num_after):
    # Load Movie Object
    video=cv2.VideoCapture(movie_filename)
    success, image=video.read()

    # Make Space for captured frames
    stack = np.zeros((image.shape[0], image.shape[1], num_before+num_after+1), dtype= np.float32)

    # Fill Stack with the frame at each event
    count = 0
    for f in range(-num_before, num_after+1):
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, event_frame+f)
        success, image = video.read()
        gray = np.mean(image, axis=2)
        stack[:,:, count] = gray
        print f
        count = count + 1
    # Release video handle
    video.release()
    return stack



    
# Extract the Frames from a movie indicated by an "Events" list and save to a numpy Stack
def frames_for_each_event_to_stack(movie_filename, events_list, offset):
    offset=0
    # Load Movie Object
    video=cv2.VideoCapture(movie_filename)
    success, image=video.read()

    # Make Space for captured frames
    stack = np.zeros((image.shape[0], image.shape[1], np.size(events_list)), dtype= np.float32)

    # Fill Stack with the frame at each event
    count = 0
    for e in events_list:
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, e+offset)
        success, image = video.read()
        gray = np.mean(image, axis=2)
        stack[:,:, count] = gray
        print e
        count = count + 1
    # Release video handle
    video.release()
    return stack

# Compute background and subtract it from a stack of frames
def remove_background(frame_stack):
    # Compute background (median along 3rd axis)
    background = np.median(frame_stack, axis=2)
    num_frames = np.size(frame_stack, axis=2)
    
    # Make space for back_sub frames
    bkg_subtracted_stack = np.zeros(np.shape(frame_stack), dtype= np.float32)

    # Sbtract background from every frame
    for i in range(0, num_frames):
        bkg_subtracted_stack[:,:,i] = frame_stack[:,:,i] - background
    return bkg_subtracted_stack

# -------------------------------------------------------------
# Return largest (area) cotour from contour list
def get_largest_contour(contours):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    if max_area > 0:
        return best_cnt, max_area
    else:
        return cnt, max_area


# 2-D Matrix Rotation
def rotation_2D(points, theta):

   R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

   return np.dot(points, R.T)

# --

# Specify parameters (filenames, etc.)
base_path = r'C:/Users/lorenza/Desktop/'
movie_filename = base_path + r'Clip00.avi'

# Extract spaced frames across the movie
offset = 0
background_stack = frames_for_each_event_to_stack(movie_filename, range(1212), offset)

# Compute background image from the "ball contact" stack
background = np.median(background_stack, axis=2)
background = np.uint8(background)





# Track nose background
plt.figure()
video=cv2.VideoCapture(movie_filename)
for i in range(0,1212):
    
    # Load new frame
    video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    success, image = video.read()
    frame = np.uint8(np.mean(image, axis=2))

    # Allocate Memory
    mask = np.zeros(frame.shape,np.uint8)
    thresholded = np.zeros(frame.shape,np.uint8)
    opened = np.zeros(frame.shape,np.uint8)

    # Subtract background
    back_sub = cv2.absdiff(frame, background)

    # Threshold Frame
    boolean = back_sub > 15
    thresholded = np.uint8(boolean)
    
    # Open (Erode then Dialate) to remove tail
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # Find Binary Contours            
    contours,hierarchy = cv2.findContours(np.copy(opened),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    # Get largest particle
    largest_cnt, area = get_largest_contour(contours)

    # Find it's orientation    
    (x,y),(MA,ma),angle = cv2.fitEllipse(largest_cnt)
    angle_radians = 2*np.pi*(angle/360.0)

    # Get centroid (and moments)          
    M = cv2.moments(largest_cnt)
    cx = (M['m10']/M['m00'])
    cy = (M['m01']/M['m00'])
    
    # Find Binary Contours            
    #contours,hierarchy = cv2.findContours(thresholded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    # Get largest particle
    #largest_cnt, area = get_largest_contour(contours)
    
    # Create Binary Mask Image (1 for Fish, 0 for Background)
    cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
    pixel_points = np.float32(np.transpose(np.nonzero(mask)))
    centered_points = np.vstack((pixel_points[:,0]-y,pixel_points[:,1]-x)).T
                        
    # Rotate Pixel Points by Angle of Major Axis (ellipse)
    rotated_points = rotation_2D(centered_points, angle_radians)
    offset_points = np.vstack((rotated_points[:,0]-y,rotated_points[:,1]-x)).T

    # Find Extreme Points in Y
    topmost = offset_points[:,0].argmin()
    bottommost = offset_points[:,0].argmax()
    
    tip1 = pixel_points[topmost,:]
    tip2 = pixel_points[bottommost,:]
    
    plt.subplot(2,2,1)
    plt.cla()
    plt.imshow(frame)
    plt.plot(tip1[1], tip1[0], 'bo', markersize=5)
    plt.plot(tip2[1], tip2[0], 'bo', markersize=5)
    plt.axis('image')
    plt.subplot(2,2,2)
    plt.plot(tip1[1], tip1[0], 'bo', markersize=5)
    plt.plot(tip2[1], tip2[0], 'bo', markersize=5)
    plt.axis('image')
    plt.subplot(2,2,3)
    plt.cla()
    plt.imshow(mask+thresholded)
    plt.plot(tip1[1], tip1[0], 'go', markersize=5)
    plt.plot(tip2[1], tip2[0], 'ro', markersize=5)
    plt.axis('image')
    plt.subplot(2,2,4)
    plt.cla()
    ax = plt.gca()
    if(i == 0):
        ax.invert_yaxis()
    plt.plot(centered_points[:,1], centered_points[:,0],'b.')
    plt.plot(rotated_points[:,1], rotated_points[:,0],'r.')
    save_filename = base_path + '\\analysis\\frames\\' + str(i) +'.png'
    plt.savefig(save_filename)
    
    print num_frames-i

# Release video handle
video.release()

# FIN

## X,Y CENTROID BALL (EASY AND LESS PRECISE WAY)
#frame_number = 41
#
#img=back_sub_stack[:,:,frame_number] # select one image from the background stack
#img = img[:300,:]# Crop out reflection
#
## Add a border around the image to avoid over shooting the matrix when cropping around ball
#border_size = 100
#width = np.size(img, axis=1)
#height = np.size(img, axis=0)
#
## Make borders (fill with zero)
#border_top = np.zeros((border_size, width))
#border_side = np.zeros((height+border_size*2, border_size))
#
## Add to tops and sides of image
#img = np.vstack((border_top, img, border_top))
#img = np.hstack((border_side, img, border_side))
#
## Normalize image to between -1 and 1 (this is required by scikit) - just divide by largest abs value
#norm_img = img/np.max(np.abs(img))
#
#threshold=norm_img>0.2 # ball is white on back background (binary img but still small white pixels all around
#ball_only=skimage.morphology.remove_small_objects(threshold, min_size=200)
#blob=skimage.feature.blob_dog(ball_only) # in this case it is more accurate but I do not
#blob = blob[0] # select forst blob
#
## CROP IMAGE (you don't need PIL...there is a numpy way to crop) 
#ballX = blob[1]
#ballY = blob[0]
#
#aligned = img[ballY-border_size:ballY+border_size, ballX-border_size:ballX+border_size]
##with 100 Gioia is half, with 200 the tail is missing, need 250, what if I am close to the edges of the frame?
#
#plt.imshow(aligned)
#
## Here you could now run the above through a loop and make a big stack...or try to save them...but
## this will also be hard given that the "aligned" images are floats...I would put them in a numpy stack..
#
#
