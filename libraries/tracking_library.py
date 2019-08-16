# -*- coding: utf-8 -*-
"""
Tracking

Tracking functions for Pac-Rat videos

@author: LCARK
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
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
        print(angle)

        # Get centroid (and moments)          
        M = cv2.moments(largest_cnt)
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])

        # Display
        display = np.hstack((threshed, opened))
        cv2.imshow("Display", display)
        cv2.waitKey(1)

    # Cleanup
    cv2.destroyAllWindows()

    return 1


#FIN