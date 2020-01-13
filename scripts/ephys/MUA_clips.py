# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:22:32 2020

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
#from filters import *
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import parser_library as prs
import behaviour_library as behaviour
import seaborn as sns
import cv2


### Load pre-processed data

rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 

Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
sessions_subset = Level_2_post



# Specify paths
session  = sessions_subset[1]
session_path =  os.path.join(hardrive_path,session)
mua_path = os.path.join(session_path +'/MUA_250_to_2000.bin')



results_dir = os.path.join(session_path +'/Mua_Clips')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
    
csv_dir_path = os.path.join(session_path + '/events/')
trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)

clips_path = os.path.join(session_path + '/Clips/')
clip = os.path.join(clips_path + 'Clip022.avi')



# Probe from superficial to deep electrode, left side is shank 11 (far back)
probe_map=np.array([[103,78,81,118,94,74,62,24,49,46,7],
                    [121,80,79,102,64,52,32,8,47,48,25],
                    [123,83,71,104,66,84,38,6,26,59,23],
                    [105,69,100,120,88,42,60,22,57,45,5],
                    [101,76,89,127,92,67,56,29,4,37,9],
                    [119,91,122,99,70,61,34,1,39,50,27],
                    [112,82,73,97,68,93,40,3,28,51,21],
                    [107,77,98,125,86,35,58,31,55,44,14],
                    [110,113,87,126,90,65,54,20,2,43,11],
                    [117,85,124,106,72,63,36,0,41,15,16],
                    [114,111,75,96,116,95,33,10,30,53,17]])

flatten_probe = probe_map.flatten()

# Load MUA (binned to frames)
mua_flat_f32 = np.fromfile(mua_path, dtype=np.float32)
mua_channels = np.reshape(mua_flat_f32, (121,-1))
mua = np.reshape(mua_channels, (11,11,-1))

# Compute full movie median (as baseline)
mua_median = np.median(mua, 2)

# Compute full movie stdev (to z-score)
mua_std = np.std(mua, 2)

# Subtract median (zero baseline) and divide by std (z-score)
mua_zeroed = np.zeros(np.shape(mua))
mua_z_score = np.zeros(np.shape(mua))
for r in range(11):
    for c in range(11):
        mua_zeroed[r,c,:] = (mua[r,c,:] - mua_median[r,c])
        mua_z_score[r,c,:] = (mua[r,c,:] - mua_median[r,c]) / mua_std[r,c]



#trial prior end to current trial end
        
ends = trial_idx[:,1]
            
trial_lenght_end_to_end = np.diff(np.hstack((0, ends)))

start_clip = ends[21]
end_clip = trial_lenght_end_to_end[22]




video=cv2.VideoCapture(clip)
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Save "movie around event"
for i in range(start_clip,start_clip + end_clip ):
  
    success, image = video.read()
    
    fig = plt.figure(figsize=(7,12))
    a = fig.add_subplot(2, 1, 1)
    imgplot = plt.imshow(image)
    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ax = fig.add_subplot(2, 1, 2)
    ax = plt.imshow(mua_zeroed[:, :, i], cmap="viridis", vmin=-2, vmax=7)
    plt.colorbar()

    plt.savefig(results_dir +'/Clip%d.png' %i)
    plt.close('all')
    
    
video.release()










