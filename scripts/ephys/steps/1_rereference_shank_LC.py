# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: median/mean rereferencing per shank

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
include_channels = np.delete(np.arange(128), exlcude_channels)

# Retreive probe map
probe_map = ephys.get_probe_map()

# Measure shank means and medians
shank_means = np.zeros((11, num_samples))
shank_medians = np.zeros((11, num_samples))
for sh in range(1,10,1):
    shank_channels = probe_map[(sh-1):(sh+1),sh]
    shank_means[sh,:] = np.mean(raw_Z[shank_channels,:], axis=0)
    shank_medians[sh,:] = np.median(raw_Z[shank_channels,:], axis=0)

# Measure probe mean and median
probe_mean = np.mean(raw_Z[include_channels,:], axis=0)
probe_median = np.median(raw_Z[include_channels,:], axis=0)

# Rereference each channel using its shank mean/median
clean_Z = np.zeros(raw_Z.shape)
for sh in range(11):
    shank_channels = probe_map[:,sh]
    for ch in shank_channels:
        raw_Z_ch = raw_Z[ch, :]
        clean_Z_ch = raw_Z_ch - probe_mean
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

#FIN
