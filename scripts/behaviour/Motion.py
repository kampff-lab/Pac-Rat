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



motion_path = 'F:/Videogame_Assay/AK_33.2/2018_04_24-17_02/motion.csv'
video_path = 'F:/Videogame_Assay/AK_33.2/2018_04_24-17_02/Video.csv'
ball_path = 'F:/Videogame_Assay/AK_33.2/2018_04_24-17_02/events/BallOn.csv'

motion = np.genfromtxt(motion_path,dtype=float) 
video = behaviour.timestamp_CSV_to_pandas(video_path)
ball = behaviour.timestamp_CSV_to_pandas(ball_path)

ball_idx = behaviour.closest_timestamps_to_events(video,ball)


before = 240
after = 360

n = len(ball_idx)
motion_around_ball = [[] for _ in range(n)] 
                       

for i, idx in enumerate(ball_idx):
    motion_around_ball[i] = motion[idx-before:idx+after]
    plt.figure()
    plt.plot(motion[idx-before:idx+after])
    
    
motion_array = np.array(motion_around_ball) 

plt.plot(motion_array[70], alpha= 0.6)
plt.ylim(0,0.1)
plt.vlines(240,0,n,'k')


plt.plot(np.mean(motion_array,axis= 0), color='k')
plt.ylim(0,0.1)
plt.vlines(240,0,n,'k')