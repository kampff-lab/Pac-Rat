# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:44 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
#from filters import *
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs
import behaviour_library as behaviour
### Load and pre-process data

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

rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post
session= sessions_subset[2]

num_channels = 128
freq = 30000
lowcut = 250
highcut = 2000


for session in sessions_subset:
    try:

        session_path = os.path.join(hardrive_path, session )
        save_path = os.path.join(session_path + '/MUA_250_to_2000.bin')
        recording_path = os.path.join( session_path + '/Amplifier_cleaned.bin')
        # - use read-only mode "r+" to prevent overwriting the original file

        samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
        samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)


        binned_signal = np.zeros((121,len(samples_for_frames)))

        for ch, channel in enumerate(flatten_probe):
            
            data = np.memmap(recording_path, dtype = np.uint16, mode = 'r')
            num_samples = int(int(len(data))/num_channels)
            recording_time_sec = num_samples/freq
            recording_time_min = recording_time_sec/60
            reshaped_data = np.reshape(data,(num_samples,128))
            #to have 128 rows
            reshaped_data_T= reshaped_data.T
            data = None
            reshaped_data = None
        
            # Extract data chunk for single channel
            channel_data = reshaped_data_T[channel,:]
            reshaped_data_T = None
        
            # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
            channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
            channel_data = None
            
            # FILTERS (one ch at the time)
            #channel_data_highpass = highpass(channel_data_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
            #data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')
            channel_data_MUA_bandpass = butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order = 3, btype = 'bandpass')
            
            #lowcut = 500
            #highcut = 2000
            #channel_data_bandpass =  butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order=3, btype='bandstop')    
            
            # Determine high and low threshold
            abs_channel_data_MUA = np.abs(channel_data_MUA_bandpass)
            
            
            sample_diff = np.diff(samples_for_frames)
            sample_diff = np.hstack((sample_diff,250))
        
            for s in np.arange(len(samples_for_frames)):
                sample = samples_for_frames[s]
                signal_to_bin = abs_channel_data_MUA[sample:(sample + sample_diff[s])]
                avg = np.mean(signal_to_bin)
                binned_signal[ch][s] = avg
                
            print(ch)
            
        mua_array = np.float32(binned_signal)
        mua_array.tofile(save_path)
        
    except Exception: 
        print('error'+ session)
    continue





#events of interest





mua_path = os.path.join(session_path +'/MUA_250_to_2000.bin')  
touch_path = os.path.join(session_path +  '/events/RatTouchBall.csv')
reward_path = os.path.join(session_path +  '/events/TrialEnd.csv')
ball_on_path = os.path.join(session_path +  '/events/BallON.csv')
video_csv = os.path.join(session_path + '/Video.csv')

video_time = behaviour.timestamp_CSV_to_pandas(video_csv)
touch_time = behaviour.timestamp_CSV_to_pandas(touch_path)
reward_time = behaviour.timestamp_CSV_to_pandas(reward_path)
ball_time = behaviour.timestamp_CSV_to_pandas(ball_on_path)


touching_light = behaviour.closest_timestamps_to_events(video_time, touch_time)
reward = behaviour.closest_timestamps_to_events(video_time, reward_time)
ball_on = behaviour.closest_timestamps_to_events(video_time, ball_time)



binned_signal_to_reshape = np.fromfile(mua_path, dtype=np.float32)
binned_signal = np.reshape(binned_signal_to_reshape, (121,-1))

events_list = [touching_light,reward,ball_on]

events = events_list[2]

offset = 360
avg_MUA_around_event = [[] for _ in range(121)] 

#plt.figure()
for count in np.arange(121):
    
    binned_signal_ch = binned_signal[count]
    try:
       
        ch_MUA = [[] for _ in range(len(events))]
        valid = []
    
        for e, event in enumerate(events):
            ch_MUA[e] = binned_signal_ch[(event-offset):(event+offset)]
            
            if np.max(ch_MUA[e]) < 40:
                valid.append(e)
#                if count == 52:  
#                    plt.plot(ch_MUA[e] + (e*10), alpha =0.5)
#                    plt.vlines(360, 0,( e*10), 'r')
                    
        ch_MUA_array = np.array(ch_MUA)
        ch_MUA_valid = ch_MUA_array[np.array(valid),:]
        avg_MUA_around_event[count]= np.mean(ch_MUA_valid, axis=0)
                
    except Exception:             
        continue





avg_MUA_around_event_array= np.array(avg_MUA_around_event)


norm_MUA_around_event = [[] for _ in range(121)] 

for count in np.arange(121):
    
    avg_MUA_around_event_ch = avg_MUA_around_event_array[count]
    median_MUA = np.median(avg_MUA_around_event_ch)
    norm_MUA_around_event[count] = avg_MUA_around_event_ch-median_MUA
    
    
norm_MUA_around_event_array = np.array(norm_MUA_around_event)    
    




# Plot raster
plt.figure()
plt.vlines(360, -1.0, len(range(121)), 'r')
for index, mua in enumerate(norm_MUA_around_event_array):   
    plt.plot(mua + (index), alpha = 0.5)

plt.plot(np.mean(norm_MUA_around_event_array[0:121:11, :], axis=0),'b')
plt.plot(np.mean(norm_MUA_around_event_array[5:121:11, :], axis=0), 'c')
plt.plot(np.mean(norm_MUA_around_event_array[10:121:11, :], axis=0), 'r')
#plt.plot(np.mean(norm_MUA_around_event_array[], axis=0))





fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.6,0.8])
image = avg_MUA_around_event_array
i = ax.imshow(image, aspect='auto', interpolation='gaussian')
colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
fig.colorbar(i, cax=colorbar_ax)
ax.vlines(360, 0, len(range(120)), 'r')


np.savetxt(r'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test2.csv',test, delimiter=',', fmt='%1.3f')

 'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test.csv'


# Save as binary file
test_f32 = np.float32(test)
save_path = r'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test_f32.bin'
test_f32.tofile(save_path)

# Open from binary file
retest_f32 = np.fromfile(save_path, dtype=np.float32)
reshaped_f32 = np.reshape(retest_f32, (121,-1))



#data_test = open('C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test.bin',encoding='utf-16', mode = 'r')






