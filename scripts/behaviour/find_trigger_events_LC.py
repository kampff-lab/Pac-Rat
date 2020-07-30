# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:17:42 2020

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import pandas as pd
import scipy
from scipy import math


#attempt to find trigger events in moving light and joystick

    


ball_tracking_path = 'F:/Videogame_Assay/AK_49.1/2019_09_20-14_18/events/BallTracking.csv'
ball_position_path ='F:/Videogame_Assay/AK_49.1/2019_09_20-14_18/events/BallPosition.csv'
trial_idx_path='F:/Videogame_Assay/AK_49.1/2019_09_20-14_18/events/Trial_idx.csv'


# parse ball position file

filename = ball_position_path
split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
timestamps = split_data[:,0]
positions_strings = split_data[:,1]
for index, s in enumerate(positions_strings):
    tmp = s.replace('(', '')
    tmp = tmp.replace(')', '')
    tmp = tmp.replace('\n', '')
    tmp = tmp.replace(' ', '')
    positions_strings[index] = tmp
positions = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
len(positions)

#open trials idx file

trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int) 

#select start (c:0) and touch(c:2) event from trial.ifx file 

start = trial_idx[:,0]
touch  = trial_idx[:,2]


diff_touch_start = np.abs(start-touch)

#remove too short trials and select good ball positions and timestamps

good_trials = diff_touch_start>10
len(good_trials)


if len(good_trials)<len(positions):
    positions = positions[:-1]
    timestamps = timestamps[:-1]

len(positions)

good_ball = positions[good_trials] #69 good trials
good_times = timestamps[good_trials]

#parse ball tracking file 

filename = ball_tracking_path
split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
timestamps = split_data[:,0]
positions_strings = split_data[:,1]
for index, s in enumerate(positions_strings):
    tmp = s.replace('(', '')
    tmp = tmp.replace(')', '')
    tmp = tmp.replace('\n', '')
    tmp = tmp.replace(' ', '')
    positions_strings[index] = tmp
positions = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
len(positions)


#loop to select all the idx where the ball coordinates in the ball position file matches with the ball coordinate in the tracking file
#rounding can be doen better? ideally choosing ceil or floor?

timestamps_good = [[] for _ in range(len(good_ball))]


for ball in range(len(good_ball)):
    
    ball_times = []
    
    for pos in range(len(positions)):
    
        if round(positions[pos,0],6) - round(good_ball[ball,0],6) == 0  and round(positions[pos,1],6) -  round(good_ball[ball,1],6) == 0:
            
            ball_times.append(pos)
        
            
        timestamps_good[ball]= ball_times

    print(ball)


#selecting the last idx which correspond to the frame before the ball starts moving. 
#ideally adding a +1 to each frame?

trigger=[]

for i in range(len(timestamps_good)):
    
    if timestamps_good[i] ==[]:
        trigger.append([])
        
    else:       
    
        trigger.append(timestamps_good[i][-1]) #62 trials detected


final_trigger = []

for t in np.array(trigger):
    final_trigger.append(t+1) # not allowed until I get rid of the empty arrays




######################################################################
#timestamp route?
        
#useful fx

#do something to the timestamps so that I can use them

def timestamp_CSV_to_pandas(filename):
    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None, usecols=[0])
    timestamp = timestamp_csv[0]
    timestamp_Series= pd.to_datetime(filename)
    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
    return timestamp_Series


#find where the closest timestamps of an event of interest timestamp is and it return the idx 
def closest_timestamps_to_events(timestamp_list, event_list):
    nearest  = []
    for e in event_list:
        delta_times = timestamp_list-e
        nearest.append(np.argmin(np.abs(delta_times)))
    return nearest  



####################################################################

time_good_ball =  pd.to_datetime(good_times) 
time_tracking = pd.to_datetime(timestamps) 
        
nearest_good_ball= closest_timestamps_to_events(time_tracking, time_good_ball)


    