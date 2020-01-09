# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:44 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from filters import *
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs
import behaviour_library as behaviour

### Load pre-processed data

rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post





# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)
mua_path = os.path.join(session_path +'/MUA_250_to_2000.bin')

save_path = os.path.join(hardrive_path, session_path +'/mua')

# Probe from superficial to deep electrode, left side is shank 11 (far back)
probe_map=np.array([[103,78,81,118,94,74,62,24,49,46,7],
                    [121,80,79,102,64,52,32,8,47,48,25],
                    [123,83,71,104,66,84,38,6,26,59,23],
                    [105,69,100,120,88,42,60,22,57,45,5],
                    [101,76,89,127,92,67,56,29,4,37,9],
                    [119,91,122,99,70,61,34,1,39,50,27],
                    [112,82,73,97,68,93,40,3,28,51,21],
                    [107,77,98,125,86,35,58,31,55,44,14],
                    [110,113,87,126,90,65,54,20,2,43,11],
                    [117,85,124,106,72,63,36,0,41,15,16],
                    [114,111,75,96,116,95,33,10,30,53,17]])

flatten_probe = probe_map.flatten()

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

# Plot z-score avg
mua_avg = np.mean(mua_z_score, 2)

# Get events of intererest indices
touch_path = os.path.join(session_path +  '/events/RatTouchBall.csv')
reward_path = os.path.join(session_path +  '/events/TrialEnd.csv')
ball_on_path = os.path.join(session_path +  '/events/BallOn.csv')
video_csv = os.path.join(session_path + '/Video.csv')
trial_start_path = os.path.join(session_path +  '/events/TrialStart.csv')




video_time = behaviour.timestamp_CSV_to_pandas(video_csv)
touch_time = behaviour.timestamp_CSV_to_pandas(touch_path)
reward_time = behaviour.timestamp_CSV_to_pandas(reward_path)
ball_time = behaviour.timestamp_CSV_to_pandas(ball_on_path)
trial_time = behaviour.timestamp_CSV_to_pandas(trial_start_path)




touching_light = behaviour.closest_timestamps_to_events(video_time, touch_time)
reward = behaviour.closest_timestamps_to_events(video_time, reward_time)
ball_on = behaviour.closest_timestamps_to_events(video_time, ball_time)
start = behaviour.closest_timestamps_to_events(video_time, trial_time)

events_list = [touching_light,reward,ball_on,start,ball_noticed]

# Average around event
events = events_list[0]
mua_event0_avg = np.mean(mua_zeroed[:, :, events], 2)

events = events_list[1]
mua_event1_avg = np.mean(mua_zeroed[:, :, events], 2)

events = events_list[2]
mua_event2_avg = np.mean(mua_zeroed[:, :, events], 2)


# Display
plt.figure()
plt.subplot(1,3,1)
plt.imshow(mua_event0_avg, vmin=-1.0, vmax=10.0)
plt.subplot(1,3,2)
plt.imshow(mua_event1_avg, vmin=-1.0, vmax=10.0)
plt.subplot(1,3,3)
plt.imshow(mua_event2_avg, vmin=-1.0, vmax=10.0)
plt.show()

# Save "movie around event"
for i in range(-240, 600):
    # Average around event (shifted by i)
    events = np.array(events_list[0]) + i
    mua_event0_avg = np.mean(mua_zeroed[:, :, events], 2)

    events = np.array(events_list[1]) + i
    mua_event1_avg = np.mean(mua_zeroed[:, :, events], 2)

    events = np.array(events_list[2]) + i
    mua_event2_avg = np.mean(mua_zeroed[:, :, events], 2)

    events = np.array(events_list[2]) - 20000 + i
    mua_event3_avg = np.mean(mua_zeroed[:, :, events], 2)

    # Create figure
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(mua_event0_avg, vmin=-1.0, vmax=12.0)
    plt.subplot(2,2,2)
    plt.imshow(mua_event1_avg, vmin=-1.0, vmax=12.0)
    plt.subplot(2,2,3)
    plt.imshow(mua_event2_avg, vmin=-1.0, vmax=12.0)
    plt.subplot(2,2,4)
    plt.imshow(mua_event3_avg, vmin=-1.0, vmax=12.0)

    # Save figure
    figure_path = save_path + '/event_avg_' + str(i+1000) + '.png'
    plt.savefig(figure_path)
    plt.close('all')




events_list = [touching_light,reward,ball_on,start]

# Average around event
events = events_list[0]
mua_event0_avg = np.mean(mua_zeroed[:, :, events], 2)

events = events_list[1]
mua_event1_avg = np.mean(mua_zeroed[:, :, events], 2)

events = events_list[2]
mua_event2_avg = np.mean(mua_zeroed[:, :, events], 2)






event_1 = np.array(events_list[0])
event_2 = np.array(events_list[1])

#
##if the lenght of the trial start is longer remove the last value 
#if len(event_1)>len(event_2):
#    event_1 = event_1[:-1]

event_diff = abs(event_2 - event_1)

final_array= np.zeros((11,11,len(event_1)))


#i=15 has a problem it gives empty array.

for i, idx  in enumerate(event_1):
    
    test_avg = mua_zeroed[:, :, idx:idx+ event_diff[i]]
    avg = np.nanmean(test_avg,2)
    final_array[:,:,i]=avg



final_avg = np.nanmean(final_array,2)




label_ball_noticed =  'D:/ShaderNavigator/annotations/AK_33.2/2018_04_29-15_43/Video.csv'

label = np.genfromtxt(label_ball_noticed,dtype=str,delimiter = ',', usecols = 0)

ball_noticed = np.genfromtxt(label_ball_noticed,dtype=int,delimiter = ',', usecols = 1)


yes_trials = []

for l,lab in enumerate(label):
    if lab == 'yes':
        yes_trials.append(l)


test_frames = frame[yes_trials]



event_1_test= event_1[yes_trials]
event_2_test= event_2[yes_trials]



#ball on to noticed


event_1 = np.array(events_list[4])
event_2 = np.array(events_list[0])

event_1 = event_1[yes_trials]
event_2 = event_2[yes_trials]





#
##if the lenght of the trial start is longer remove the last value 
#if len(event_1)>len(event_2):
#    event_1 = event_1[:-1]

event_diff = abs(event_2 - event_1)

final_array= np.zeros((11,11,len(event_1)))


#i=15 has a problem it gives empty array.

for i, idx  in enumerate(event_1):
    
    test_avg = mua_zeroed[:, :, idx:idx+ event_diff[i]]
    avg = np.nanmean(test_avg,2)
    final_array[:,:,i]=avg



final_avg = np.nanmean(final_array,2)









#noticed to touch












