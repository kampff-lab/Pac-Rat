# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:27:08 2019

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm  
from matplotlib.colors import LogNorm 
from pylab import *


hardrive_path = r'F:/' 
figure_name = 'RAT_AK_33.2_centroid_tracking.png'
plot_main_title = 'RAT AK_33.2 Centroid'


sessions_subset = Level_2_pre
number_of_subplots= len(sessions_subset)



#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
f=plt.figure(figsize=(20,10))
f.suptitle(plot_main_title)


   
for i,session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f.add_subplot(2, 4, 1+i, frameon=False)
        ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(0.3))
    except Exception: 
        print (session + '/error')
        continue       


f.tight_layout()
f.subplots_adjust(top = 0.87)
        

















