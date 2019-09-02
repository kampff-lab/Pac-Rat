# -*- coding: utf-8 -*-
"""
Tracking

Tracking functions for Pac-Rat videos

@author: LCARK
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Compute background (median)
def compute_background_median(video, start, stop, step):

    # Read current frame
    success, image = video.read()

    # Measure dimensions
    width = image.shape[1]
    height = image.shape[0]
    depth = int(np.floor((stop-start) / step))

    # Make an empty background stack
    stack = np.zeros((height, width, depth), dtype=np.uint8)

    # Fill background stack
    count = 0
    for frame in range(start, stop, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stack[:,:, count] = gray
        count = count + 1

    # Compuet median background frame
    background = np.median(stack, axis=2)

    return np.uint8(background)

def rotation_2D(points, theta):

   R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

   return np.dot(points, R.T)

# Get largest contour
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

# Track rat through entire video
def track_rat(video, background):

    # Read current frame
    success, image = video.read()

    # Measure dimensions
    width = image.shape[1]
    height = image.shape[0]
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Reset video to first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read and process each frame
    cv2.namedWindow("Display")
    for frame in range(0, num_frames):

        # Read current frame
        success, image = video.read()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Subtract background (darker values positive)
        subtracted = cv2.subtract(background, gray)

        # Threshold
        level, threshed = cv2.threshold(subtracted, 5, 255, cv2.THRESH_BINARY)

        # Open (Erode then Dialate) to remove tail and noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
        mask = np.zeros(image.shape,np.uint8)
        
        # Find Binary Contours            
        ret, contours, hierarchy = cv2.findContours(np.copy(opened),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)        
        if len(contours) == 0:
            # Store a NaN in the tracking thing that doesn't exist yet
            continue

        # Get largest particle
        largest_cnt, area = get_largest_contour(contours)

        # Find it's orientation    
        (x,y),(MA,ma),angle = cv2.fitEllipse(largest_cnt)
        angle_radians = 2*np.pi*(angle/360.0)
        #print(angle)

        # Get centroid (and moments)          
        M = cv2.moments(largest_cnt)
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])
        
        
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
        plt.imshow(subtracted)
        plt.subplot(2,2,2)
        plt.cla()
        plt.imshow(opened)
        plt.plot(tip1[1], tip1[0], 'go', markersize=5)
        plt.plot(tip2[1], tip2[0], 'ro', markersize=5)
        plt.axis('image')
        plt.subplot(2,2,3)
        plt.cla()
        plt.imshow(mask)
        plt.plot(tip1[1], tip1[0], 'go', markersize=5)
        plt.plot(tip2[1], tip2[0], 'ro', markersize=5)
        plt.axis('image')
        plt.subplot(2,2,4)
        plt.cla()
        plt.plot(centered_points[:,0], centered_points[:,1],'b.')
        plt.plot(rotated_points[:,0], rotated_points[:,1],'r.')
        
        
        
        
        
        
        #diff= cv2.absdiff(threshed,opened)

        
        # Display
        #display = np.hstack((diff, opened_diff))
        #cv2.imshow("Display",largest_cnt)
        #cv2.waitKey(1)

    # Cleanup
    cv2.destroyAllWindows()

    return 1

# Crop rat through entire video
def crop_rat(video, background, window_size, output_name):

    # Create output filenames
    csv_path = output_name + '.csv'
    avi_path = output_name + '.avi'

    # Open output video

    # Compute crop size (half of window size)
    crop_size = np.int(window_size / 2)

    # Read current frame
    success, image = video.read()

    # Measure dimensions
    width = image.shape[1]
    height = image.shape[0]
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create "larger" frame to contain video frame with border
    container = np.zeros((height + 2 * crop_size, width + 2 * crop_size), dtype=np.uint8)

    # Reset video to first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read and process each frame
    cv2.namedWindow("Display")
    for frame in range(0, num_frames):

        # Read current frame
        success, image = video.read()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Subtract background (darker values positive)
        subtracted = cv2.subtract(background, gray)

        # Threshold
        level, threshed = cv2.threshold(subtracted, 5, 255, cv2.THRESH_BINARY)

        # Open (Erode then Dialate) to remove tail and noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)

        # Find Binary Contours            
        ret, contours, hierarchy = cv2.findContours(np.copy(opened),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)        
        if len(contours) == 0:
            # Store a NaN in the tracking thing that doesn't exist yet
            continue

        # Get largest particle
        largest_cnt, area = get_largest_contour(contours)

        # Get centroid (and moments)          
        M = cv2.moments(largest_cnt)
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])

        # Insert grayscale frame into the container
        container[crop_size:(height+crop_size), crop_size:(width+crop_size)] = gray

        # Offset centroid position to container coordinates
        container_cx = np.int(cx + crop_size)
        container_cy = np.int(cy + crop_size)

        # Crop around the centroid position in the container image
        crop = container[(container_cy - crop_size):(container_cy + crop_size), (container_cx - crop_size):(container_cx + crop_size)]
        
        # Write output video frame

        # Display
        cv2.imshow("Display", crop)
        cv2.waitKey(1)

    # Cleanup
    cv2.destroyAllWindows()
    # Clouse output video

    return 1

#FIN