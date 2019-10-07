# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:27:08 2019

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import behaviour_library as behaviour
import parser_library as prs
from matplotlib.colors import PowerNorm  
from matplotlib.colors import LogNorm 
from pylab import *


hardrive_path = r'F:/' 
rat_ID = 'AK_33.2'
figure_name = figure_name = 'RAT_' + rat_ID + '_Centroid_tracking.pdf'
plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1'

rat_summary_table_path = r'F:/Videogame_Assay/AK_33.2_Pt.csv'



Level_1 = prs.Level_1_paths(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
Level_3_moving = prs.Level_3_moving_light_paths(rat_summary_table_path)



sessions_subset = Level_2_pre


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
f=plt.figure(figsize=(20,10))
f.suptitle(plot_main_title)


   
for i, session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f.add_subplot(2, 4, 1+i, frameon=False)
        ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis')
        ax.set_title('session %d' %i, fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f.tight_layout()
f.subplots_adjust(top = 0.87)
        


#CREATING A FOLDER CALLED 'SUMMARY' IN THE MAIN RAT FOLDER AMD SAVING THE FIG IN FORMAT .tiff



#main folder rat ID
script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + rat_ID)
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f.savefig(results_dir + figure_name, transparent=True)
#f.savefig(results_dir + figure_name)      
    
















