# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:58:37 2019

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
from correlations_def import *

# GENERATING EVENT LISTS
base_folder = r'F:'
rat_ID=r'/AK_40.2/'

rat_folder = base_folder + rat_ID
sessions = os.listdir(rat_folder)

# setting offset around event

#pre_offset = 150000
#post_offset = 
#window_size = pre_offset
#pre_offset_samples = '150000'

N=121
impedance_th = 5.0*1e6 # MOmh

folder_name = "_Correlations"


for session in sessions[:9]:
    try:
        # Load Files Needed


        session_path = rat_folder + session + r'/'
        events_folder_path = os.path.join(session_path, r'events/')
        trial_end_file = events_folder_path + 'TrialEnd.csv'  
        touch_ball_file = events_folder_path + 'RatTouchBall.csv'
        amplifier_file = session_path + 'Amplifier.bin'
        #sync_file = session_path + 'Sync.bin'
        video_csv = session_path + 'Video.csv'
        samples_for_frames_file_path = session_path + r'Analysis/samples_for_frames.csv'
        impedance_file_path = session_path + 'impedance1.csv'
        
        
        
        day = rat_folder + session
        #create a folder where to save the plots
        folder = os.path.join(day,'Ephys_Analysis/')
        results_dir = os.path.join(folder, 'Correlations/')
        
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        
        

        #offset_list_names = [trial_end_file, touch_ball_file]
        offset_list_names = [trial_end_file]
        for offset_list_name in offset_list_names:
            offset_list = event_finder(offset_list_name, video_csv, samples_for_frames_file_path)
            corr_highpass_before = correlation_avg_highpass_before_event(results_dir,amplifier_file, offset_list, event = 'reward_tone', window_size = 150000, highpass_filter = butter_filter, highpass_lowcut = 500, highpass_highcut = 1000)
            corr_lowpasss_before = correlation_avg_lowpass_before_event(results_dir,amplifier_file, offset_list, event = 'reward_tone', window_size = 150000 ,lowpass_filter = butter_filter_lowpass,lowpass_lowcut = 250)
            corr_highpass_after = correlation_avg_highpass_after_event(results_dir,amplifier_file, offset_list, event = 'reward_tone', window_size = 150000, highpass_filter = butter_filter, highpass_lowcut = 500, highpass_highcut = 1000)
            corr_highpass_after = correlation_avg_lowpass_after_event(results_dir,amplifier_file, offset_list, event = 'reward_tone', window_size = 150000, lowpass_filter = butter_filter_lowpass, lowpass_lowcut = 250)







        #find bad channels using impedance.csv
        
        #imp_file = pd.read_csv(impedance_file_path)
        #imp = imp_file['Impedance Magnitude at 1000 Hz (ohms)']
        #imp_array=np.array(imp)
        #impedance = imp_array.astype(dtype=int)
        
        #map the impedance
        
        #probe_map_as_vector = np.reshape(probe_map.T, newshape = N)
        
        #impedance_mapped = impedance[probe_map_as_vector]
        
        #bad_channels = np.where(impedance_mapped > impedance_th)[0]


        #find the missed trial idx list, make it into an array, multiply by 2 and substract the 
        #the index number
        #missed_idx = idx_Event(trial_end_file)
        #missed_idx_array = np.array(missed_idx)
        #missed_lenght_list = list(range(len(missed_idx)))
        #missed_idx_to_remove = missed_idx_array * 2 - missed_lenght_list
        #missed_idx_to_remove_list = missed_idx_to_remove.tolist()

        #Remove missed trials (automatic)
        #for index in sorted(missed_idx, reverse=True):
        #   del closest_reward_tone[index]


        
        print(session_path+'session_DONE')
        print(len(offset_list))
        
    except Exception:
        print(session_path+'something_WRONG')
        continue
        

























    
    
    
    
## Step 1 - Make a scatter plot with square markers, set column names as labels
#
#def heatmap(x, y, size):
#    fig, ax = plt.subplots()
#    
#    # Mapping from column names to integer coordinates
#    x_labels = [v for v in sorted(x.unique())]
#    y_labels = [v for v in sorted(y.unique())]
#    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
#    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
#    
#    size_scale = 500
#    ax.scatter(
#        x=x.map(x_to_num), # Use mapping for x
#        y=y.map(y_to_num), # Use mapping for y
#        s=size * size_scale, # Vector of square sizes, proportional to size parameter
#        marker='s' # Use square as scatterplot marker
#    )
#    
#    # Show column labels on the axes
#    ax.set_xticks([x_to_num[v] for v in x_labels])
#    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
#    ax.set_yticks([y_to_num[v] for v in y_labels])
#    ax.set_yticklabels(y_labels)
#    
#data = pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')
#columns = ['bore', 'stroke', 'compression-ratio', 'horsepower', 'city-mpg', 'price'] 
#corr = data[columns].corr()
#corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
#corr.columns = ['x', 'y', 'value']
#heatmap(
#    x=corr['x'],
#    y=corr['y'],
#    size=corr['value'].abs()
#)
#
#ax.grid(False, 'major')
#ax.grid(True, 'minor')
#ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
#ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
#
#
#
#ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
#ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

