# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:47:14 2020

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 

# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)

# Test ephys quality and pre-processing on test clips from prior Trial end to current Trial end 

# Specify paths
session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

# Recording data path
raw_recording = os.path.join(session_path +'/Amplifier.bin')
cleaned_recording = os.path.join(session_path +'/Amplifier_cleaned.bin')
mua_path = os.path.join(session_path +'/MUA_250_to_2000.bin')

# Specify sample range for clip
start_sample = 32837802
num_samples = 657331

# Specify channel of interest
depth = 6
shank = 10

# Load raw data
raw_uV = ephys.get_channel_raw_clip_from_amplifier(raw_recording, depth, shank, start_sample, num_samples)
mean_raw_ch = np.mean(raw_uV)
median_raw_ch = np.median(raw_uV)

# Load cleaned data
cleaned_uV = ephys.get_channel_raw_clip_from_amplifier(cleaned_recording, depth, shank, start_sample, num_samples)

# Remove 50 Hz
lowcut= 40
highcut= 60

wo50_iir = ephys.iirnotch_50(raw_uV, fs=30000, quality=30)
wo50_butter = ephys.butter_bandstop(raw_uV,lowcut, highcut, fs=30000, order=3)

# Plot
plt.plot(raw_uV[:150000], 'r', alpha = 0.4)
plt.plot(cleaned_uV[:150000], 'g', alpha = 0.4)
plt.plot(wo50[:150000], 'b', alpha = 0.4)
plt.show()

# Plot residuals
plt.plot(raw_uV[:150000] - wo50[:150000], 'k', alpha = 1.0)
plt.show()




#remove 50
lowcut= 48
highcut= 52

wo50 = ephys.butter_bandstop(raw_uV,lowcut, highcut, fs=30000, order=3, btype='bandstop')
plt.plot(wo50[:150000],alpha = 0.4)


raw_diff = raw_uV - wo50
cleaned_diff = cleaned_uV - wo50

plt.plot(raw_diff[:150000],alpha = 0.4)
plt.plot(cleaned_diff[:150000],alpha = 0.4)


# highpass

highpass_cleaned = ephys.highpass(wo50,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
plt.plot(highpass_cleaned[:150000],alpha = 0.4)





#lowpass
lowcut = 250

lowpass_cleaned = ephys.butter_filter_lowpass(wo50,lowcut, fs=30000, order=3, btype='lowpass')
plt.plot(lowpass_cleaned[:150000],alpha = 0.4)






#bandpass

lowcut=
hightcut = 

bandpass_cleaned = ephys.butter_bandpass(wo50,lowcut, highcut, fs=30000, order=3, btype='bandpass')










## Plot
#for ch, channel in enumerate(flatten_probe):
#    plt.plot((ch*1000) + np.float32(chunk_data[channel, :]))
#plt.show()




#####################MUA#################




#trial prior end to current trial end to use with MUA which are binned based on the frames 
        
ends = trial_idx[:,1]
            
trial_lenght_end_to_end = np.diff(np.hstack((0, ends)))

start_clip = ends[21]
end_clip = trial_lenght_end_to_end[22]




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



# declaring magnitude of repetition 
K = 4
  
# using itertools.chain.from_iterable()  
# + itertools.repeat() repeat elements K times 

new_acce = np.empty((3, 156453120))

for row, aux in enumerate(accel_channels):
    test = accel_channels[row]
    
    res = list(itertools.chain.from_iterable(itertools.repeat(i, K) for i in test)) 
    new_acce[row,:]=res



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