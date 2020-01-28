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

# Load Data as uint16 from binary file, use memory mapping (i.e. do not load into RAM)
#   - use read-only mode "r+" to prevent overwriting the original file
filename = 'F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/Amplifier.bin'
num_channels = 128
data = np.memmap(filename, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
reshaped_data_T= reshaped_data.T
data = None
reshaped_data = None

# Extract data chunk for single channel
channel = 1
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

plt.figure()
#plt.plot(channel_data_bandpass[2000000:3100000])
plt.plot(channel_data_uV[2000000:3100000])
#plt.plot(data_zero_mean[55][100000:105000])
plt.title('channel_' + str(channel))



# RASTER CODE

# Determine high and low threshold
abs_channel_data_highpass = np.abs(channel_data_highpass)
sigma_n = np.median(abs_channel_data_highpass) / 0.6745
#sigma_n = np.std(abs_channel_data_highpass)
spike_threshold_hard = -5.0 * sigma_n
spike_threshold_soft = -3.0 * sigma_n

# Find spikes (peaks between high and low threshold crossings)
spike_start_times = []
spike_stop_times = []
spiking = False

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

# Get event times
event_file ='F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/events/RatTouchBall.csv'
video_csv ='F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/Video.csv'
samples_for_frames_file_path ='F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/Analysis/samples_for_frames.csv'
touching_light = event_finder(event_file,video_csv,samples_for_frames_file_path)

event_file ='F:/Videogame_Assay/AK_33.2/2018_04_29-15_43/events/TrialEnd.csv'
reward_tone = event_finder(event_file,video_csv,samples_for_frames_file_path)
annotation_path ='D:/ShaderNavigator/annotations/AK_33.2/2018_04_29-15_43/Video.csv'
annotation_str = np.genfromtxt(annotation_path, delimiter=',', usecols=0, dtype= str)

yes_idx = []
for idx, word in enumerate(annotation_str):
    if word =='yes':
        yes_idx.append(idx)
        
annotation_idx = np.genfromtxt(annotation_path, delimiter=',', usecols=1, dtype= int)

yes = annotation_idx[yes_idx]

sample_for_each_video_frame = np.genfromtxt(samples_for_frames_file_path, delimiter=',', usecols=0, dtype=np.uint32)
event_list = sample_for_each_video_frame[yes]

event_list = touching_light
event_list = reward_tone

# Make lists of spikes surrounding each event
spikes_around_event= []


for event in event_list:
    try:
        min_range = event - 150000
        max_range = event + 150000
        spike_list =[]
        for peak in peak_times:
            if (peak > min_range) and (peak < max_range):
                spike_list.append(peak - event)    
        spikes_around_event.append(spike_list)
    except Exception:
        continue
    


# Plot raster
plt.figure()
plt.vlines(0, 0, len(spikes_around_event), 'r')
for index, spikes in enumerate(spikes_around_event):
    plt.vlines(spikes,index,index+1, color=[0,0,0,0.1])
    


#FIN