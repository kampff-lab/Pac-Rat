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

hardrive_path = r'F:/'

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']



RAT_ID = RAT_ID_ephys[0] #[0]
rat_summary_table_path=rat_summary_ephys[0]#[0]



for r, rat in enumerate(rat_summary_table_path[1:]): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)

    
    for s, session in enumerate(sessions_subset[2:]):        
       
        
        session_path =  os.path.join(hardrive_path,session)



        # Specify session folder
        #session_path =  'F:/Videogame_Assay/AK_33.2/2018_04_29-15_43'#'/home/kampff/Dropbox/LCARK/2018_04_29-15_43'
        #session_path =  '/media/kampff/Data/Dropbox/LCARK/2018_04_29-15_43'
        
        # Specify data paths
        data_path = os.path.join(session_path +'/Amplifier_cleaned.bin')
        
        # Detect spikes on each channel (using weak thresholds for picking up many spikes)
        ephys.detect_MUA(data_path)
        MUA_path = data_path[:-4] + '_MUA.npz'
        
        # Label MUA (valid vs artefact)
        ephys.label_MUA(MUA_path)
        
        # Bin MUA (and resmaple to 1 kHz)
        ephys.bin_MUA(MUA_path)#raw_MUA_path
        
        print(session)
        
        

#open binned file 
        
        
        
        
        
data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(down))/num_raw_channels)

binned_mua_path = data_path[:-4] +'__BINNED.bin'
binned_mua_raw = fromfile(binned_mua_path, dtype=uint8)
binned_mua_reshape =  np.reshape(binned_mua_raw,(int(len(binned_mua_raw)/128),128)).T

samples_diff = num_samples-(len(binned_mua_raw)/128)
print(samples_diff)




plt.figure()
start = 60000
stop = 590000
plt.imshow(binned_mua_reshape[:, start:stop], aspect='auto')
plt.show()        
        
        
        
        
        
        
        