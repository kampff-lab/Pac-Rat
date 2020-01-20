# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:38:26 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import os
from scipy.signal import butter, filtfilt#, lfilter
import scipy.signal as signal
import pandas as pd

# Load and pre-process Data
# -----------------------------------------------------------------------------

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
#filename = 'F:/AK_40.2/2018_11_27-14_16/Amplifier.bin'

def GET_data_zero_mean_remapped_window(filename, offset, num_samples):
    
    num_channels = 128
    bytes_per_sample = 2
    offset_position = offset * num_channels * bytes_per_sample
    
    # Open file and jump to offset position
    f = open(filename, "rb")
    f.seek(offset_position, os.SEEK_SET)

    # Load data from this file position
    data = np.fromfile(f, dtype=np.uint16, count=(num_channels * num_samples))
    f.close()
    
    # Reshape data
    reshaped_data = np.reshape(data,(num_samples,128)).T
    #to have 128 rows
    
    # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
    data_uV = (reshaped_data.astype(np.float32) - 32768) * 0.195
    
    # Subtract channel mean from each channel
    mean_per_channel_data_uV = np.mean(data_uV,axis=1,keepdims=True)
    data_zero_mean = data_uV - mean_per_channel_data_uV
    
    # Extract (remapped) 121 probe channels
    probe_map_as_vector = np.reshape(probe_map.T, newshape=(121))
    data_zero_mean_remapped = data_zero_mean[probe_map_as_vector,:]
    
    return data_zero_mean_remapped




# FILTERS IN USE 



def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)


def butter_lowpass(lowcut, fs=30000, order=3, btype='lowpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype=btype)
    return b, a


def butter_filter_lowpass(data, lowcut,  fs=30000, order=3, btype='lowpass'):
    b, a = butter_lowpass(lowcut, fs, order=order, btype=btype)
    y = filtfilt(b, a, data)
    return y

def butter_filer_lowpass(lowcut, fs=30000, order=3, btype='lowpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype=btype)
    y = filtfilt(b, a, data)
    return y
    








def butter_filter(data, lowcut, highcut, fs=30000, order=3, btype='bandstop'):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs=30000, order=3, btype='bandstop'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


# it takes the TrialEnd.csv file in the rat folder and look for the missed trilas and gives a list of idx
    
def idx_Event(trial_end):
    RewardOutcome_file=np.genfromtxt(trial_end,usecols=[1], dtype= str)
    RewardOutcome_idx=[]
    count=0
    for i in RewardOutcome_file:
        count += 1
        if i =='Missed':
            RewardOutcome_idx.append(count-1)
    reward=np.array(RewardOutcome_idx)
    return RewardOutcome_idx





def idx_Event(trial_end):
    RewardOutcome_file=np.genfromtxt(trial_end,usecols=[1], dtype= str)
    RewardOutcome_idx=[]
    for count, in enumerate(RewardOutcome_file):
        if e =='Missed':
            RewardOutcome_idx.append(i)
    reward=np.array(RewardOutcome_idx)
    return RewardOutcome_idx





def tone_frame(target_dir,video_avi_file_path,nearest):
    video=cv2.VideoCapture(video_avi_file_path)
    success, image=video.read()
    success=True
    count = 0
    for i in nearest:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = video.read()
        if count < 10:
            cv2.imwrite(os.path.join(target_dir,"frame0%d.jpg" %count), image)
        else:
            cv2.imwrite(os.path.join(target_dir,"frame%d.jpg" %count), image)
        count += 1
    return image


def closest_value_in_array(array,value_list):
    nearest  = []
    for e in value_list:
        delta = array-e
        nearest.append(np.argmin(np.abs(delta)))
    return nearest   



def timestamp_CSV_to_pandas(filename):
    timestamp_csv = pd.read_csv(filename, delimiter=' ',header=None,usecols=[0])
    timestamp = timestamp_csv[0]
    timestamp_Series= pd.to_datetime(timestamp)
    #timestamp_csv=pd.read_csv(reward, header = None,usecols=[0],parse_dates=[0])
    return timestamp_Series

      
def closest_timestamps_to_events(timestamp_list, event_list):
    nearest  = []
    for e in event_list:
        delta_times = timestamp_list-e
        nearest.append(np.argmin(np.abs(delta_times)))
    return nearest  


def event_finder(event_file,video_csv,samples_for_frames_file_path):
    
    event_time = timestamp_CSV_to_pandas(event_file)
    video_time = timestamp_CSV_to_pandas(video_csv)
    closest_event = closest_timestamps_to_events(video_time, event_time)
    sample_for_each_video_frame = np.genfromtxt(samples_for_frames_file_path, delimiter=',', usecols=0, dtype=np.uint32)
    event_list = sample_for_each_video_frame[closest_event]
    offset_list = event_list
    return offset_list



def  correlation_avg_highpass_before_event(results_dir, amplifier_file, offset_list, event = 'touch', window_size = 150000, highpass_filter = butter_filter, highpass_lowcut = 500, highpass_highcut = 1000):
    
    for i in range(len(offset_list)):
        corr_matrix2 = np.zeros((121,121),dtype=float)
        data_zero_mean_remapped = GET_data_zero_mean_remapped_window(amplifier_file, offset_list[i] - window_size, window_size)
        all_data_zero_mapped_w50 = np.apply_along_axis(butter_filter, lowcut = 48, highcut = 52, btype='bandstop', arr = data_zero_mean_remapped, axis = 1)
        data_w50_highpass = np.apply_along_axis(highpass_filter,  lowcut=highpass_lowcut,  highcut=highpass_highcut, btype='bandpass', arr = all_data_zero_mapped_w50, axis = 1)#need t oput a minus 
        
      
        for e in range(0, window_size, 30):
            outer_product = np.outer(data_w50_highpass[:, e], data_w50_highpass[:, e])
            corr_matrix2 = corr_matrix2 + outer_product
        corr_matrix2 = corr_matrix2 / window_size / 30  
        
        norm_corr_matrix2 = np.zeros((121,121),dtype = float)
        for r in range(121):
            for c in range(121):
                normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
                norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor
        plt.figure()
        mask = np.zeros_like(norm_corr_matrix2, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right", shrink=0.5),cmap="bwr",vmax=1, vmin=-0.5, mask = mask, center = 0)#vmax=1, vmin=-0.5
        trials_dir = os.path.join(results_dir, event + '_Correlation_highpass_before/'.format(filecount=i))
            
        if not os.path.isdir(trials_dir):
            os.makedirs(trials_dir)
            
        plt.savefig(trials_dir +"correlation_highpass_before{filecount}.png".format(filecount=i))
        plt.close('all')
        print("Current offset: " + str(i))    


def  correlation_avg_lowpass_before_event(results_dir,amplifier_file, offset_list, event = 'touch', window_size = 150000 ,lowpass_filter = butter_filter_lowpass,lowpass_lowcut = 250):

    for i in range(len(offset_list)):
        corr_matrix2 = np.zeros((121,121),dtype=float)
        data_zero_mean_remapped = GET_data_zero_mean_remapped_window(amplifier_file, offset_list[i] - window_size, window_size)
        all_data_zero_mapped_w50 = np.apply_along_axis(butter_filter, lowcut = 48, highcut = 52, btype='bandstop', arr = data_zero_mean_remapped, axis = 1)
        data_w50_lowpass = np.apply_along_axis(lowpass_filter , lowcut = lowpass_lowcut , arr = all_data_zero_mapped_w50, axis = 1)
      
        for e in range(0, window_size, 30):
            
            outer_product = np.outer(data_w50_lowpass[:, e], data_w50_lowpass[:, e])
            corr_matrix2 = corr_matrix2 + outer_product
        corr_matrix2 = corr_matrix2 / window_size / 30  
        
        norm_corr_matrix2 = np.zeros((121,121),dtype = float)
        for r in range(121):
            for c in range(121):
                normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
                norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor
        plt.figure()
        mask = np.zeros_like(norm_corr_matrix2, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right", shrink=0.5),cmap="bwr",vmax=1, vmin=-0.5, mask = mask, center = 0)#vmax=1, vmin=-0.5
        trials_dir = os.path.join(results_dir, event + '_Correlation_lowpass_before_/'.format(filecount=i))
            
        if not os.path.isdir(trials_dir):
            os.makedirs(trials_dir)
            
        plt.savefig(trials_dir +"correlation_lowpass_before{filecount}.png".format(filecount=i))
        plt.close('all')
        print("Current offset: " + str(i))
        
        
def correlation_avg_highpass_after_event(results_dir,amplifier_file, offset_list, event = 'touch', window_size = 150000, highpass_filter = butter_filter, highpass_lowcut = 500, highpass_highcut = 1000):

    for i in range(len(offset_list)):
        corr_matrix2 = np.zeros((121,121),dtype=float)
        data_zero_mean_remapped = GET_data_zero_mean_remapped_window(amplifier_file, offset_list[i] + window_size, window_size)
        all_data_zero_mapped_w50 = np.apply_along_axis(butter_filter, lowcut = 48, highcut = 52, btype='bandstop', arr = data_zero_mean_remapped, axis = 1)
        data_w50_highpass = np.apply_along_axis(highpass_filter, lowcut = highpass_lowcut, highcut = highpass_highcut, btype='bandpass', arr = all_data_zero_mapped_w50, axis = 1)#need t oput a minus 
        
      
        for e in range(0, window_size, 30):
            #outer_product = np.outer(data_zero_mean_remapped[:, e], data_zero_mean_remapped[:, e])
            outer_product = np.outer(data_w50_highpass[:, e], data_w50_highpass[:, e])
            corr_matrix2 = corr_matrix2 + outer_product
        corr_matrix2 = corr_matrix2 / window_size / 30  
        
        norm_corr_matrix2 = np.zeros((121,121),dtype = float)
        for r in range(121):
            for c in range(121):
                normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
                norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor
        plt.figure()
        mask = np.zeros_like(norm_corr_matrix2, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right", shrink=0.5),cmap="bwr",vmax=1, vmin=-0.5, mask = mask, center = 0)#vmax=1, vmin=-0.5
        trials_dir = os.path.join(results_dir, event + '_Correlation_highpass_after/'.format(filecount=i))
            
        if not os.path.isdir(trials_dir):
            os.makedirs(trials_dir)
            
        plt.savefig(trials_dir +"correlation_highpass_after{filecount}.png".format(filecount=i))
        plt.close('all')
        print("Current offset: " + str(i))



def  correlation_avg_lowpass_after_event(results_dir,amplifier_file, offset_list, event = 'touch', window_size = 150000, lowpass_filter = butter_filter_lowpass, lowpass_lowcut = 250):

    for i in range(len(offset_list)):
        corr_matrix2 = np.zeros((121,121),dtype=float)
        data_zero_mean_remapped = GET_data_zero_mean_remapped_window(amplifier_file, offset_list[i] + window_size, window_size)
        all_data_zero_mapped_w50 = np.apply_along_axis(butter_filter, lowcut = 48, highcut = 52, btype='bandstop', arr = data_zero_mean_remapped, axis = 1)
        data_w50_lowpass = np.apply_along_axis(butter_filter_lowpass , lowcut = lowpass_lowcut , arr = all_data_zero_mapped_w50, axis = 1)
      
        for e in range(0, window_size, 30):
            
            outer_product = np.outer(data_w50_lowpass[:, e], data_w50_lowpass[:, e])
            corr_matrix2 = corr_matrix2 + outer_product
        corr_matrix2 = corr_matrix2 / window_size / 30  
        
        norm_corr_matrix2 = np.zeros((121,121),dtype = float)
        for r in range(121):
            for c in range(121):
                normalization_factor = (corr_matrix2[r,r] + corr_matrix2[c,c])/2
                norm_corr_matrix2[r,c] = corr_matrix2[r,c]/normalization_factor
        plt.figure()
        mask = np.zeros_like(norm_corr_matrix2, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ax = sns.heatmap(norm_corr_matrix2, cbar_kws = dict(use_gridspec = False,location = "right", shrink=0.5),cmap="bwr",vmax=1, vmin=-0.5, mask = mask, center = 0)#vmax=1, vmin=-0.5
        trials_dir = os.path.join(results_dir, event + '_Correlation_lowpass_after_/'.format(filecount=i))
            
        if not os.path.isdir(trials_dir):
            os.makedirs(trials_dir)
            
        plt.savefig(trials_dir +"correlation_lowpass_after{filecount}.png".format(filecount=i))
        plt.close('all')
        print("Current offset: " + str(i))

