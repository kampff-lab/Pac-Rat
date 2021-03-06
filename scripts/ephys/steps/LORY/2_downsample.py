# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: downsample to 1 kHz

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

rat_summary_table_path = 'F:/Videogame_Assay/AK_40.2_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)


# Specify data path
data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')

# Downsample data 
raw = ephys.downsample_amplifier(data_path)

# Load and display downsampled data
ch = 24
cleaned = np.fromfile(data_path, count=(30000*30*128), dtype=np.uint16)
cleaned = np.reshape(cleaned, (-1, 128)).T
downsampled_path = data_path[:-4] + '_downsampled.bin'
downsampled = np.fromfile(downsampled_path, count=(1000*30*128), dtype=np.uint16)
downsampled = np.reshape(downsampled, (-1, 128)).T
plt.plot(cleaned[ch,:], 'r')
plt.plot(np.arange(30000) * 30, downsampled[ch,:], 'b')
plt.show()

#FIN