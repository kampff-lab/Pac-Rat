# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1a: (headtsage) mean rereferencing/cleaning for MUA analysis

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

# Specify session folder
session_path =  '/home/kampff/Dropbox/LCARK/2018_04_29-15_43'

# Specify raw data path
raw_path = os.path.join(session_path +'/Amplifier.bin')

# Specify the disconnected channels (Intan Channel #)
disconnected_channels = np.array([12, 13, 18, 19, 108, 109 ,115])

# Determine the bad channels to exclude (in reference calculation)
#bad_channels_idx = ephys.bad_channel(session_path, min_imp = 10000, max_imp = 6000000)
#bad_channels_path =  os.path.join(session_path+'/bad_channels.csv')
#bad_channels = np.genfromtxt(bad_channels_path, delimiter=',',dtype=int)
            
#bad channels are idx while disconnecte are the actual channel number, for the cleaning I need the actual channel number to sepa
##rate between headstages
##exclude_channels = np.sort(np.hstack((disconnected_channels,probe_map_flatten[bad_channels])))
exclude_channels = disconnected_channels
print(len(exclude_channels))

# Clean raw data and store binary file
ephys.clean_raw_amplifier(raw_path, exclude_channels)

# Load cleaned data and display
data = np.fromfile(raw_path, count=(30000*60*128), dtype=np.uint16)
data = np.reshape(data, (-1, 128)).T
plt.plot(data[21,:], 'r')
cleaned_path = raw_path[:-4] + '_cleaned.bin'
data = np.fromfile(cleaned_path, count=(30000*60*128), dtype=np.uint16)
data = np.reshape(data, (-1, 128)).T
plt.plot(data[21,:], 'b')
plt.show()

#FIN
