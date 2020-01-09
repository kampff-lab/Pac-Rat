# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:44 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs
from scipy import signal
import itertools 

## Load accelerometer data

rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post

# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)
accel_path = os.path.join(session_path +'/Accelerometer.bin')
recording_path =  os.path.join(session_path +'/Amplifier.bin')
touching_path =  os.path.join(session_path +'/events/RatTouchBall.csv')
frame_to_sample_path =  os.path.join(session_path +'/Analysis/samples_for_frames.csv')
video_csv_path = os.path.join(session_path +'/Video.csv')
ball_path = os.path.join(session_path + '/events/BallOn.csv')

num_channels = 128
data = np.memmap(recording_path, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)

# Load accelerometer data (7500 Hz)
accel_flat_u16 = np.fromfile(accel_path, dtype=np.uint16)
accel_channels = np.reshape(accel_flat_u16, (-1,3)).T

# Plot
plt.figure()
plt.plot(accel_channels[:,180000:210000].T)
plt.show()

# Upsample to 30,000 Hz (4x)

#b = signal.resample(accel_channels[0,:], len(accel_channels) * 4)
#b_int = b.astype(int)


  
# declaring magnitude of repetition 
K = 4
  
# using itertools.chain.from_iterable()  
# + itertools.repeat() repeat elements K times 

new_acce = np.empty((3, 156453120))

for row, aux in enumerate(accel_channels):
    test = accel_channels[row]
    
    res = list(itertools.chain.from_iterable(itertools.repeat(i, K) for i in test)) 
    new_acce[row,:]=res


plt.figure()
plt.plot(new_acce[:,180000:210000].T)


touch_in_samples = event_finder(touching_path,video_csv_path,frame_to_sample_path)
ball_in_samples = event_finder(ball_path,video_csv_path,frame_to_sample_path)


events = [ball_in_samples,touch_in_samples]

event = events[0]


before = 240
after = 240

n = len(event)

aux1 = [[] for _ in range(n)] 
                      
aux = new_acce[2]
    
for i, idx in enumerate(event):
    aux1[i] = aux[idx-before:idx+after]






