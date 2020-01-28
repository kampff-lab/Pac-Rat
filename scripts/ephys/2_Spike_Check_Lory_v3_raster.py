# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from filters import highpass
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')

import parser_library as prs
results_dir = 'C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/rasters'

rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post
#Level_3_post = prs.Level_3_post_paths(rat_summary_table_path)
#sessions_subset = Level_3_post
### Load and pre-process data

session = sessions_subset[2]

recording =  os.path.join(hardrive_path, session +'/Amplifier_cleaned.bin')

touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
video_csv = os.path.join(hardrive_path, session +'/Video.csv')
trial_end_path =os.path.join(hardrive_path, session +'/events/'+'TrialEnd.csv')
samples_for_frames_file_path = os.path.join(hardrive_path, session +'/Analysis/'+'samples_for_frames.csv')

sample_for_each_video_frame = np.genfromtxt(samples_for_frames_file_path, delimiter=',', usecols=0, dtype=np.uint32)

#annotation_path ='D:/ShaderNavigator/annotations/AK_33.2/2018_05_05-09_55/Video.csv'

touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
reward_tone = event_finder(trial_end_path,video_csv,samples_for_frames_file_path)
#annotation_str = np.genfromtxt(annotation_path, delimiter=',', usecols=0, dtype= str)
#annotation_idx = np.genfromtxt(annotation_path, delimiter=',', usecols=1, dtype= int)

yes_idx = []
for idx, word in enumerate(annotation_str):
    if word =='yes':
        yes_idx.append(idx)
 



       
ball_noticed = annotation_idx[yes_idx]
surprise_moment = annotation_idx
surprise_idx = sample_for_each_video_frame[surprise_moment]
ball_noticed_idx = sample_for_each_video_frame[ball_noticed]
touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
reward_tone = event_finder(trial_end_path,video_csv,samples_for_frames_file_path)


events = [touching_light,reward_tone]
event_names = ['touch','reward']


#events = [surprise] ,reward_tone,ball_noticed]
#event_names = ['surprise'] ,'reward_tone','ball_noticed'] 

# Probe from superficial to deep electrode, left side is shank 11 (far back)
probe_map = np.array([[103,78,81,118,94,74,62,24,49,46,7],
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


offset_before = 150000
offset_after = 150000
    
    

flatten_probe = probe_map.flatten()
num_channels = 128
freq = 30000



for channel in flatten_probe:
    
    data = np.memmap(recording, dtype = np.uint16, mode = 'r')
    num_samples = int(int(len(data))/num_channels)
    recording_time_sec = num_samples/freq
    recording_time_min = recording_time_sec/60
    reshaped_data = np.reshape(data,(num_samples,128))
    #to have 128 rows
    reshaped_data_T= reshaped_data.T
    data = None
    reshaped_data = None

    # Extract data chunk for single channel
    minutes = np.int(recording_time_min)
    seconds = minutes*60
    num_samples_per_chunk = seconds*freq
    channel_data = reshaped_data_T[channel,:num_samples_per_chunk]
    reshaped_data_T = None

    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
    channel_data = None
    
    # FILTERS (one ch at the time)
    channel_data_highpass = highpass(channel_data_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    #data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')
    #channel_data_highpass = butter_filter(channel_data_uV, 500, 5000, fs=30000, order=3, btype='bandpass')
    
    #lowcut = 500
    #highcut = 2000
    #channel_data_bandpass =  butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order=3, btype='bandstop')

    # RASTER CODE
    
    # Determine high and low threshold
    abs_channel_data_highpass = np.abs(channel_data_highpass)
    sigma_n = np.median(abs_channel_data_highpass) / 0.6745
    #sigma_n = np.std(abs_channel_data_highpass)
    spike_threshold_hard = -5.0 * sigma_n
    spike_threshold_soft = -3.0 * sigma_n



    # Find threshold crossings
    spike_start_times, spike_stop_times = threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft)
    
    # Find peak voltages and times
    spike_peak_voltages = []
    spike_peak_times = []
    for start, stop in zip(spike_start_times,spike_stop_times):
        peak_voltage = np.min(channel_data_highpass[start:stop]) 
        peak_voltage_idx = np.argmin(channel_data_highpass[start:stop])
        spike_peak_voltages.append(peak_voltage)
        spike_peak_times.append(start + peak_voltage_idx)
      
    # Remove too early and too late spikes
    spike_starts = np.array(spike_start_times)
    spike_stops = np.array(spike_stop_times)
    peak_times = np.array(spike_peak_times)
    peak_voltages = np.array(spike_peak_voltages)
    good_spikes = (spike_starts > 100) * (spike_starts < (len(channel_data_highpass)-200))

    # Select only good spikes
    spike_starts = spike_starts[good_spikes]
    spike_stops = spike_stops[good_spikes]
    peak_times = peak_times[good_spikes]
    peak_voltages = peak_voltages[good_spikes]
    print(channel)

    for event in np.arange(len(events)):
        event_list = events[event]
        # Make lists of spikes surrounding each event
        spikes_around_event= []
        try:
            for idx in event_list:
    
                min_range = idx - offset_before
                max_range = idx + offset_after
                spike_list =[]
                for peak in peak_times:
                    if (peak > min_range) and (peak < max_range):
                        spike_list.append(peak - idx)    
                spikes_around_event.append(spike_list)
            
            # Plot raster
            f = plt.figure()
            figure_name = '/channel_' + str(channel) + event_names[event] +'.png'
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine() 
            plt.vlines(0, 0, len(spikes_around_event), 'r')
            for index, spikes in enumerate(spikes_around_event):
                plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
                plt.title('channel_' + str(channel) + event_names[event])
                #if index == 5 :
                    #plt.vlines(spikes, index, index+1, color = [0,250,250,0.1])
                    
                #else:
                    #plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
                                            
                #plt.title('channel_' + str(channel) + event_names[event])
                #save the fig in .tiff
            f.savefig(results_dir + figure_name, transparent=False)
            plt.close(f)

        except Exception:
            continue
 



for event in np.arange(len(events)):
    event_list = events[event]
    # Make lists of spikes surrounding each event
    spikes_around_event= []
    for idx in event_list:
    
        min_range = idx - offset_before
        max_range = idx + offset_after
        spike_list =[]
        for peak in peak_times:
            if (peak > min_range) and (peak < max_range):
                spike_list.append(peak - idx)    
        spikes_around_event.append(spike_list)

    f = plt.figure(figsize=(12,5))
    figure_name = '/channel_' + str(channel) + event_names[event] +'.png'
    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine() 
    plt.vlines(0, 0, len(spikes_around_event), 'r')
    for index, spikes in enumerate(spikes_around_event):
        plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
        plt.title('channel_' + str(channel) + event_names[event])
                #if index == 5 :
                    #plt.vlines(spikes, index, index+1, color = [0,250,250,0.1])
                    
                #else:
                    #plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
                                            
                #plt.title('channel_' + str(channel) + event_names[event])
                #save the fig in .tiff
    f.savefig(results_dir + figure_name, transparent=False)



           
# Are spikes downward or upward?
def threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft):
    
    spike_start_times = []
    spike_stop_times = []
    spiking = False
    
    for i, voltage in enumerate(channel_data_highpass):
        # Look for a new spike
        if(not spiking):
            if(voltage < spike_threshold_hard):
                spiking = True
                spike_start_times.append(i)
        # Track ongoing spike            
        else:
            # Keep track of max (negative) voltage until npo longer spiking
            if(voltage > spike_threshold_soft):
                spiking = False       
                spike_stop_times.append(i)
                  
    return spike_start_times, spike_stop_times


#FIN