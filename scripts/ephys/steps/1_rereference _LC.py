# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: median/mean rereferencing

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)


#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 


rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)

#recording data path
raw_recording = os.path.join(session_path +'/Amplifier.bin')
cleaned_recording = os.path.join(session_path +'/Amplifier_cleaned.bin')
mua_path = os.path.join(session_path +'/MUA_250_to_2000.bin')


#clip of interest 
clip_number = 'Clip022.avi'
clips_path = os.path.join(session_path + '/Clips/')
clip = os.path.join(clips_path + clip_number)



#idx ro identify the start and the end of the clip of interest both in ephys samples and frames   
csv_dir_path = os.path.join(session_path + '/events/')
trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)

video_csv = os.path.join(session_path + '/Video.csv')

samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)


#trial prior end to current trial end based on ephys samples tp use with raw and cleaned recordings

end_samples = event_finder(trial_end_idx,video_csv,samples_for_frames_file_path)
samples_lenght_end_to_end = np.diff(np.hstack((0, end_samples)))
sample_start_clip = end_samples[21]
clip_sample_lenght = samples_lenght_end_to_end[22]



#Load raw data

start_sample = sample_start_clip
num_samples = clip_sample_lenght



# Specify data paths
raw_path = raw_recording

# Specify sample range for clip
start_sample = 32837802
num_samples = 657331

# Specify channel of interest
depth = 6
shank = 10

# Load raw data and convert to microvolts
raw_uV = ephys.get_raw_clip_from_amplifier(raw_path, start_sample, num_samples)

# Compute mean and standard deviation for each channel
raw_mean = np.mean(raw_uV, axis=1)
raw_std = np.std(raw_uV, axis=1)

# Z-score each channel
raw_Z = np.zeros(raw_uV.shape)
for ch in range(128):
    raw_Z[ch,:] = (raw_uV[ch,:] - raw_mean[ch]) / raw_std[ch]

# Specify channels to exclude
#exlcude_channels = np.array([12, 13, 18, 19, 108, 109 ,115])
exlcude_channels = np.array([12, 13, 18, 54, 108, 109 ,115])

# Determine channels to exclude on each headstage
A_exclude_channels = exlcude_channels[exlcude_channels < 64]
B_exclude_channels = exlcude_channels[exlcude_channels >= 64]

# Determine headstage channels
A_channels = np.arange(64)
B_channels = np.arange(64, 128)

# Remove excluded channels
A_channels = np.delete(A_channels, A_exclude_channels)
B_channels = np.delete(B_channels, B_exclude_channels)

# Compute median values for each headstage
A_median = np.median(raw_Z[A_channels,:], axis=0)
B_median = np.median(raw_Z[B_channels,:], axis=0)

# Compute mean values for each headstage
A_mean = np.mean(raw_Z[A_channels,:], axis=0)
B_mean = np.mean(raw_Z[B_channels,:], axis=0)

# Rereference each channel
clean_Z = np.zeros(raw_Z.shape)
for ch in A_channels:
    raw_Z_ch = raw_Z[ch, :]
    clean_Z_ch = raw_Z_ch - A_mean
    clean_Z[ch,:] = clean_Z_ch
for ch in B_channels:
    raw_Z_ch = raw_Z[ch, :]
    clean_Z_ch = raw_Z_ch - B_mean
    clean_Z[ch,:] = clean_Z_ch

# Plot Z-scored ephys data
plt.figure()

# cleaned
probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)
plt.subplot(1,2,1)
offset = 0
colors = cm.get_cmap('tab20b', 11)
for shank in range(11):
    for depth in range(11):
        ch = (depth * 11) + shank
        plt.plot(probe_Z[ch, 142000:155000] + offset, color=colors(shank))
        offset += 2
# raw
plt.subplot(1,2,2)
probe_Z = ephys.apply_probe_map_to_amplifier(raw_Z)
offset = 0
colors = cm.get_cmap('tab20b', 11)
for shank in range(11):
    for depth in range(11):
        ch = (depth * 11) + shank
        plt.plot(probe_Z[ch, 142000:155000] + offset, color=colors(shank))
        offset += 2
plt.show()

### LORY add threshold crossing spike counter
probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)

clean_model  = ephys.apply_probe_map_to_amplifier(clean)

raw_signal = ephys.apply_probe_map_to_amplifier(raw_uV)

shank_ref =  ephys.apply_probe_map_to_amplifier(clean_Z)
probe_no_Z = ephys.apply_probe_map_to_amplifier(clean_no_Z)


n = len(probe_Z)  

spike_times_no_Z = [[] for _ in range(n)]  

spike_times_Z = [[] for _ in range(n)]  

spike_times_clean_model = [[] for _ in range(n)] 
  
spike_times_raw = [[] for _ in range(n)] 

spike_times_shank =  [[] for _ in range(n)] 

signal = probe_no_Z


for channel in np.arange(len(probe_Z)):

    try:
    
        channel_data = signal[channel,:]
        
        # FILTERS (one ch at the time)
        channel_data_highpass = ephys.highpass(channel_data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    
        # Determine high and low threshold
        abs_channel_data_highpass = np.abs(channel_data_highpass)
        sigma_n = np.median(abs_channel_data_highpass) / 0.6745
        
        #adaptive th depending of ch noise
        spike_threshold_hard = -5.0 * sigma_n
        spike_threshold_soft = -3.0 * sigma_n
        
        #fixed th given we have Zscored the data
        
        #spike_threshold_hard_fixed = -1.5
        #spike_threshold_soft_fixed = -.5
        
        
        # Find threshold crossings
        spike_start_times, spike_stop_times = threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft)
        
        # Find threshold crossings with fixed th 
    
        #spike_start_times_fixed, spike_stop_times_fixed = threshold_crossing(channel_data_highpass,spike_threshold_hard_fixed,spike_threshold_soft_fixed)
        
        
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
        
        peak_times_corrected  = start_sample + peak_times
        #spike_times_Z[channel] = peak_times_corrected
        #spike_times_clean_model[channel] = peak_times_corrected
        #spike_times_raw[channel] = peak_times_corrected
        #spike_times_shank[channel] = peak_times_corrected
        spike_times_no_Z[channel] = peak_times_corrected
        print(channel)
        
    except Exception:
        continue
     


 # Plot raster
f = plt.figure()

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
for index, spikes in enumerate(spike_times_Z):
    plt.vlines(spikes, index, index+1, color = [0,1,0,0.1])
plt.title('headstage')

# Plot raster
f = plt.figure()

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
for index, spikes in enumerate(spike_times_raw):
    plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
plt.title('raw')   
   

f = plt.figure()

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
for index, spikes in enumerate(spike_times_clean_model):
    plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
plt.title('model')   
   

f = plt.figure()

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
for index, spikes in enumerate(spike_times_shank):
    plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
plt.title('shank')   
   


f = plt.figure()

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
for index, spikes in enumerate(spike_times_no_Z):
    plt.vlines(spikes, index, index+1, color = [0,0,0,0.1])
plt.title('no_Z')   
   















































### LATER

# Spectrogram test
plt.figure()
shank = 4
for depth in range(11):
    plt.subplot(11,2,depth*2 + 1)
    probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)
    fs = 30000
    ch = (depth * 11) + shank
    f, t, Sxx = signal.spectrogram(probe_Z[ch,:], fs, nperseg=30000, nfft=30000, noverlap=27000)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim([0, 25])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.subplot(11,2,depth*2 + 2)
    plt.plot(probe_Z[ch,:])
plt.show()










































#FIN

#    samples_fft = np.fft.rfft(probe_Z[ch,:])
#    frequencies = np.abs(samples_fft)
#    plt.plot(frequencies)
