# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 3: detect spikes on each channel

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
#os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
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


# Specify session folder
#session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
#session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)


# Specify data paths
data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')

# Detect spikes 
ephys.detect_spikes(data_path)


#loading .npz file

spike_detected = os.path.join(session_path +'/Amplifier_cleaned_spikes.npz')

data = np.load(spike_detected, allow_pickle=True)

#spike_times=spike_times, spike_peaks=spike_peaks


spike_times = data['spike_times']
spike_peaks = data['spike_times']