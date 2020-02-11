# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:11:08 2020

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


num_raw_channel = 128
num_channels = 121


for ch in np.arange(num_channels):
    
    try:
        # Extract data for single channel
        channel_data = signal_cleaned[ch,:]
        
        # FILTERS (one ch at the time)
        channel_data_highpass = ephys.highpass(channel_data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
    
        # Determine high and low threshold
        abs_channel_data_highpass = np.abs(channel_data_highpass)
        sigma_n = np.median(abs_channel_data_highpass) / 0.6745
        
        #adaptive th depending of ch noise
        spike_threshold_hard = -3.0 * sigma_n
        spike_threshold_soft = -1.0 * sigma_n
        
        # Find threshold crossings
        spike_start_times, spike_stop_times = threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft)    
            