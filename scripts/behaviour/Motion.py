# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:25:12 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import behaviour_library as behaviour

motion_stimulus_path = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/motions.csv'
motion_stimulus= np.genfromtxt(motion_stimulus_path, delimiter = ',', usecols = 1) 

video_path = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/Video.csv'
ball_path = 'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/events/BallOn.csv'

motion = np.genfromtxt(motion_stimulus_path,dtype=float,delimiter = ',', usecols = 0 ) 
video = behaviour.timestamp_CSV_to_pandas(video_path)
ball = behaviour.timestamp_CSV_to_pandas(ball_path)

ball_idx = behaviour.closest_timestamps_to_events(video,ball)



#extract motion chunk around ball on 

before = 240
after = 480

n = len(ball_idx)

motion_around_ball = [[] for _ in range(n)] 
                       

for i, idx in enumerate(ball_idx):
    motion_around_ball[i] = motion[idx-before:idx+after]
#   plt.figure()
#   plt.plot(motion_stimulus[idx-before:idx+after])
    
# = next(x[0] for x in enumerate(motion_around_ball[40]) if x[1] > 200000)


correction_idx = []

for chunk in motion_around_ball:
    idx = next(x[0] for x in enumerate(chunk) if x[1] > 200000)
    correction_idx.append(idx)
    
corrected_ball_idx = np.array(ball_idx) + np.array(correction_idx)
    
next(x[0] for x in enumerate(values) if x[1] > 0.9)    
    

motion_around_ball = motion_around_ball[1:]


plt.plot(motion_around_ball[40])
plt.vlines(8,0,1000000,'k')
    
motion_array = np.array(new_motion_around_ball) 
motion_array2 = np.array(motion_around_ball2) 

plt.figure()
plt.plot(motion_array[39], alpha= 0.4)
#plt.plot(motion_array2[25], alpha= 0.9)
plt.ylim(0,0.02)
#plt.vlines(240,0,0.2,'k', alpha= 0.5)


avg_motion_around_ball= np.mean(motion_array[1:], axis  = 0)

plt.plot(avg_motion_around_ball)
plt.ylim(0,1)
plt.vlines(240,0,1,'k', alpha= 0.5)














#plt.plot(np.mean(motion_array,axis= 0), color='k')
#plt.ylim(0,0.1)
#plt.vlines(240,0,n,'k')
#
#
#
#ball_on = []
#
#for v, value in enumerate(motion_stimulus):
#    
#    if value> 900000:
#        ball_on.append(v)
#        
#ball_sub = np.array(ball_on[1:-1]) - np.array(ball_idx)   
#    
#
#
#





    
    