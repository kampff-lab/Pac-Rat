# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 2a: measure MUA

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
#os.sys.path.append('D:/Repos/Pac-Rat/libraries')
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




rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']

RAT_ID = RAT_ID_ephys 

hardrive_path = r'F:/'


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys


for r, rat in enumerate(rat_summary_table_path): 
    
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    for s, session in enumerate(sessions_subset):        
        try:
                
            # Specify data path
            data_path = os.path.join(hardrive_path+session +'/Amplifier_cleaned.bin')
            
            # Downsample data 
            ephys.detect_MUA(data_path)
            
            # Load and display downsampled data
     
            print(session + 'saved')

        except Exception: 
            print(session+'error')
            continue   
    
    

# Specify session folder
#session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'

# Specify data paths
data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')

# Detect spikes on each channel (using weak thresholds for picking up many spikes)
ephys.detect_MUA(data_path)

# To Do...
# - Detect artifact spikes (when spikes occur on too many channels within 1 ms of one another)
# - Count spikes per video-frame (removing or setting to NaN frames with artifact spikes)
# -  - this "spike count" per frame per channel will be the MUA signal that we use for subsequent analysis 
