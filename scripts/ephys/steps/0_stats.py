# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 0: measure recording statistics

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
#os.sys.path.append('/home/kampff/Repos/Kampff-Lab/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
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




rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[0]
session_path =  os.path.join(hardrive_path,session)

#recording data path
raw_recording = os.path.join(session_path +'/Amplifier.bin')








# Specify session folder
#session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'
#
# Specify data paths
raw_path = os.path.join(session_path +'/Amplifier.bin')

# Load raw data and convert to microvolts
mean_int, std_int, mean_uV, std_uV = ephys.measure_raw_channel_stats(raw_path, 0)

#FIN