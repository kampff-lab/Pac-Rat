# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:07:20 2015

@author: lorenza
"""

# Import Required Python Librariers
import numpy as np
import pandas as pd
import cv2
import csv
import os 
import matplotlib.pyplot as plt
import glob
import math
from RingList import RingList

# Import new librariers
import skimage 
from skimage import feature
from skimage import io
from skimage import morphology
#from scipy.spatial import distance




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
        
        
        
        
def crop(bkg_stack,light_position,target_dir):
    ball_position= np.genfromtxt(light_position, dtype='int')
    ball_list=ball_position.tolist()
    ball=ball_list.pop()
    #after using pop the ball_list has been updated and I can use it because now 
    #it doesnt contain the last row
    size=100
    count=0
    x=[]
    y=[]
    listoffiles=os.listdir(bkg_stack)
    for i in listoffiles:
        img=cv2.imread(os.path.join(bkg_stack,i),cv2.IMREAD_GRAYSCALE)
        th=img>90
        img_8=np.uint8(th)
        M = cv2.moments(img_8)
        cx = (M['m10']/M['m00'])
        cy = (M['m01']/M['m00'])
        x.append(cx)
        y.append(cy)
        bkg_img_crop = img[(y[count]-size):(y[count]+size), (x[count]-size):(x[count]+size)]
        cv2.imwrite(os.path.join(target_dir,"cropped_%d.png" %count), bkg_img_crop) 
        count +=1
        plt.figure(2)
        plt.imshow(bkg_img_crop)
        #save_filename =   r'C:/Users/lorenza/Desktop/color_crntroid'+'\\' + str(count) +'.png'
        #plt.savefig(save_filename)
