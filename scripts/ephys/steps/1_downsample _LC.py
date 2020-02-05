# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: downsample to 1 kHz

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
import mne
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from mne import time_frequency
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


#idx ro identify the start and the end of the clip of interest both in ephys samples and frames   
csv_dir_path = os.path.join(session_path + '/events/')
touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)

video_csv = os.path.join(session_path + '/Video.csv')

samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)


#trial prior end to current trial end based on ephys samples tp use with raw and cleaned recordings
touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)

#end_samples = event_finder(trial_end_idx,video_csv,samples_for_frames_file_path)
#samples_lenght_end_to_end = np.diff(np.hstack((0, end_samples)))
#sample_start_clip = end_samples[21]
#clip_sample_lenght = samples_lenght_end_to_end[22]




num_channels = 128
data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
#reshaped_data_T= reshaped_data.T
data = None


signal_reshaped = ephys.apply_probe_map_to_amplifier(reshaped_data)
reshaped_data = None

# Extract data chunk for single channel
channel = 37

channel_data = signal_reshaped[channel,:]
signal_reshaped = None


#ch_mean = np.mean(channel_data, axis=0)

#ch_std = np.std(channel_data, axis=0)

#channel_data_Z = channel_data - ch_mean



# Z-score each channel



#raw_Z = np.zeros(raw_uV.shape)
#for ch in range(128):
#    raw_Z[ch,:] = (raw_uV[ch,:] - raw_mean[ch]) / raw_std[ch]

# Specify channels to exclude

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
channel_data = None





data_lowpass = butter_filter_lowpass(channel_data_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
#channel_data_highpass = butter_filter(channel_data_uV, 500, 5000, fs=30000, order=3, btype='bandpass')
plt.figure()
plt.plot(data_lowpass[30000:45000])



data_downsampled = data_lowpass[::30]

plt.figure()
plt.plot(data_downsampled[1000:1500])



#test mne fx for multitaper 
        
   

p, f = time_frequency.psd_array_multitaper(data_lowpass[15000:30000], sfreq= 30000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)

plt.figure()
plt.plot(f,p)



pd, fd = time_frequency.psd_array_multitaper(data_downsampled[500:1000], sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
plt.figure()
plt.plot(fd,pd)


offset = 3000

downsampled_touch = np.uint32(np.array(touching_light)/30)

chunk_around_event =np.zeros((len(downsampled_touch),offset*2))

for e, event in enumerate(downsampled_touch):
    try:  
        chunk_around_event[e,:] = data_downsampled[event-offset : event+offset]
        print(e)
    except Exception:
        continue
 

chunk_lenght = offset*2

p_test, f_test = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)

plt.figure()
plt.plot(f_test,p_test[1,:])


p_avg = np.mean(p_test, axis =0)
f_avg  = np.mean(f_test)



plt.figure()
plt.plot(f_test,p_avg[:])
plt.title('ch_'+str(channel))




#samples_fft = np.fft.rfft(chunk_around_event)
#frequencies = np.abs(samples_fft)
freq_mean = np.mean(frequencies, axis=0)
plt.plot(freq_mean[:100])




#f, t, Sxx = signal.spectrogram(chunk_around_event, 1000, nperseg=1000, nfft=1000, noverlap=500)

### lowpass filter LFP

lowcut = 250

lowpass_data = np.zeros((len(probe_Z),num_samples))
lowpass_downsampled = [[] for _ in range(len(probe_Z))]  

for channel in np.arange(len(probe_Z)):
    try:  
        channel_data = probe_Z[channel,:]
        lowpass_cleaned = ephys.butter_filter_lowpass(channel_data,lowcut, fs=30000, order=3, btype='lowpass')
        downsampled_ch = lowpass_cleaned[::30]
        lowpass_data[channel,:] = lowpass_cleaned
        lowpass_downsampled[channel] = downsampled_ch
        print(channel)
        
    except Exception:




        
        


 f, t, Zxx = signal.stft(lowpass_cleaned, fs, nperseg=20000)

plt.pcolormesh(t, f, np.abs(Zxx))


# Downsample each channel
num_ds_samples = np.int(np.floor(num_samples / 30))
downsampled = np.zeros((128, num_ds_samples))
for ch in range(128):
    raw_ch = raw[ch,:]
    lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
    downsampled_ch = lowpass_ch[::30]
    downsampled[ch, :] = downsampled_ch[:num_ds_samples]




lowpass_data[22,:]

# Store downsampled data in a binary file







# Report
ch = 21
raw_ch = raw[ch,:]
lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
downsampled_ch = downsampled[ch, :]
plt.figure()
plt.plot(raw_ch, 'r')
plt.plot(lowpass_ch, 'g')
plt.plot(np.arange(num_ds_samples) * 30, downsampled_ch, 'b')
plt.show()

# LORY (spectral analysis, LFP, etc.)

#FIN    
        
#2 and 65 opposite phase        
        
        
plt.plot(lowpass_data[100,:150000],alpha = 0.4)




##### downsampling from 30kHz to 1kHz






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
    plt.ylim([0, 30])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(11,2,depth*2 + 2)
    plt.plot(probe_Z[ch,:])
plt.show()

#FIN