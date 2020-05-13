# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 0: compute channel stats (mean and stdev)

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

rat_summary_table_path = 'F:/Videogame_Assay/AK_41.1_Pt.csv'
hardrive_path = r'F:/' 
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post


# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)




# Specify raw data path
raw_path = os.path.join(session_path +'/Amplifier.bin')

# Compute and store channel stats
ephys.measure_raw_amplifier_stats(raw_path)

# Load channel stats and display
stats_path = raw_path[:-4] + '_stats.csv'
stats = np.genfromtxt(stats_path, dtype=np.float32, delimiter=',')
plt.plot(stats[:,1], 'b.')
plt.show()

#FIN




