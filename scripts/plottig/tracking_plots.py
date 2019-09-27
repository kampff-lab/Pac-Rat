# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:27:08 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm  
from matplotlib.colors import LogNorm 
from pylab import *

figure_name = 'RAT_AK_33.2_centroid_tracking.png'
plot_main_title = 'RAT AK_33.2 '




f.suptitle(plot_main_title)
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=True)




#CALCULATING SUCCESS AND MISSED TRIALS PER EACH SESSION OF EACH LEVEL AND PLOT 4X4 FIG

success_trials_L_1, missed_trials_L_1 = PLOT_trial_and_misses(Level_1)

x = np.array(range(len((Level_1))))

ax[0,0].bar(x, success_trials_L_1, color ='g', edgecolor ='white', width = 1, label ='Rewarded trial', alpha = .6)
# Create green bars (middle), on top of the firs ones
ax[0,0].bar(x, missed_trials_L_1, bottom = success_trials_L_1, color ='b', edgecolor ='white', width = 1, label ='Missed trial', alpha = .6)
ax[0,0].legend(loc ='best', frameon=False , fontsize = 'x-small') #ncol=2
ax[0,0].set_title('Level 1', fontsize = 13)
ax[0,0].set_ylabel('Trials / Session', fontsize = 10)
#ax[0,0].set_xlabel('Sessions')




sessions_subset = Level_1



      
        
plt.hist2d(x_centroid, y_centroid, bins=150, norm=LogNorm())

def PLOT_rat_cetroid(sessions_subset):   
    plt.figure(figsize(10,20))
    number_of_subplots= len(sessions_subset)
    for session in sessions_subset: 
        for i,v in enumerate(range(number_of_subplots)):
            centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
            centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
            centroid_tracking_wo_nan= centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
            x_centroid = centroid_tracking_wo_nan[:,0]
            y_centroid = centroid_tracking_wo_nan[:,1]    
            v = v +1 
            ax1 = subplot(number_of_subplots,1,v)
            ax1.hist2d(x_centroid, y_centroid, bins=150, norm=LogNorm())#norm=PowerNorm(0.3)
        

count =1
session = sessions_subset[1]

for i,v in enumerate(xrange(number_of_subplots)):
    v = v+1
    ax1 = subplot(number_of_subplots,1,v)
    ax1.plot(x,y)

