# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: (headtsage) mean rereferencing

@author: KAMPFF-LAB-ANALYSIS3
"""
# -*- coding: utf-8 -*-
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

rat_summary_table_path = 'F:/Videogame_Assay/AK_41.1_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[5]
session_path =  os.path.join(hardrive_path,session)

# Specify raw data path
raw_path = os.path.join(session_path +'/Amplifier.bin')

# Specify channels to exclude (in reference calculation)
#exclude_channels = np.array([12, 13, 18, 54, 108, 109 ,115])
exclude_channel_to_sort= np.sort([12, 13, 18, 54, 108, 109 ,115,19,103,  24,  49,  46, 102,  32,   8,  47, 104,  26,  22,  57,  45,
       101,  29,   4,  37,  61,   1,  39,  40,   3,  28,  51,  31,  55,
        44, 110, 113,  20,   2,  43, 106,  72,   0,  41,  15,  96,  10,
        30,  53])

exclude_channels=exclude_channel_to_sort
# Clean raw data and store binary file
ephys.clean_raw_amplifier(raw_path, exclude_channels)   

# Load cleaned data and display
data = np.fromfile(raw_path, count=(30000*3*128), dtype=np.uint16)
data = np.reshape(data, (-1, 128)).T
plt.plot(data[24,:], 'r')
cleaned_path = raw_path[:-4] + '_cleaned.bin'
data = np.fromfile(cleaned_path, count=(30000*3*128), dtype=np.uint16)
data = np.reshape(data, (-1, 128)).T
plt.plot(data[24,:], 'b')
plt.show()


#FIN

