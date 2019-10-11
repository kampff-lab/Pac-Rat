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
from matplotlib.ticker import LogFormatterExponent

rat_summary_table_path = 'F:/Videogame_Assay/AK_49.2_behaviour_only.csv'
hardrive_path = r'F:/' 
rat_ID = 'AK_49.2'


Level_0 = prs.Level_0_paths(rat_summary_table_path)
Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat_summary_table_path)
Level_1_10000 = prs.Level_1_paths_10000(rat_summary_table_path)
Level_1_20000 = prs.Level_1_paths_20000(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)
#Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
#Level_3_moving = prs.Level_3_moving_light_paths(rat_summary_table_path)



########################PLOTS#####################################




figure_name0 = figure_name = 'RAT_' + rat_ID + '_Centroid_tracking_Level_0.pdf'
plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_0'

sessions_subset = Level_0


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
f0 =plt.figure(figsize=(20,10))
f0.suptitle(plot_main_title)


   
for i, session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f0.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f0.tight_layout()
f0.subplots_adjust(top = 0.87)

##############################################################################


figure_name1 = figure_name = 'RAT_' + rat_ID + '_Centroid_tracking_Level_1_6000_3000.pdf'

plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1_6000_3000'

sessions_subset = Level_1_6000_3000


number_of_subplots= len(sessions_subset)


#f,ax = plt.subplots(2,4,figsize=(20,10),sharex=True, sharey=True)
f1 =plt.figure(figsize=(20,10))
f1.suptitle(plot_main_title)


   
for i, session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f1.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f1.tight_layout()
f1.subplots_adjust(top = 0.87)


#####################################################################################       



figure_name2 = figure_name = 'RAT_' + rat_ID + '_Centroid_tracking_Level_1_10000.pdf'
sessions_subset = Level_1_10000

number_of_subplots= len(sessions_subset)

plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1_10000'

f2 =plt.figure(figsize=(20,10))
f2.suptitle(plot_main_title)


   
for i, session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f2.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f2.tight_layout()
f2.subplots_adjust(top = 0.87)
        



#############################################################################################

figure_name3 = figure_name = 'RAT_' + rat_ID + '_Centroid_tracking_Level_1_20000.pdf'
sessions_subset = Level_1_20000

number_of_subplots= len(sessions_subset)

plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_1_20000'

f3 =plt.figure(figsize=(20,10))
f3.suptitle(plot_main_title)


   
for i, session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f3.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f3.tight_layout()
f3.subplots_adjust(top = 0.87)



##########################################################################################

figure_name4 = 'RAT_' + rat_ID + '_Centroid_tracking_Level_2_pre.pdf'
sessions_subset = Level_2_pre

number_of_subplots= len(sessions_subset)

plot_main_title = 'RAT ' + rat_ID + ' Centroid_' + 'Level_2_pre'

f4 =plt.figure(figsize=(20,10))
f4.suptitle(plot_main_title)


   
for i, session in enumerate(sessions_subset): 
    try:
        centroid_tracking_path = os.path.join(hardrive_path, session + '/crop.csv')
        centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
        centroid_tracking_wo_nan = centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
        x_centroid = centroid_tracking_wo_nan[:,0]
        y_centroid = centroid_tracking_wo_nan[:,1]    
        ax = f4.add_subplot(2, 4, 1+i, frameon=False)
        plot = ax.hist2d(x_centroid, y_centroid, bins=150, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)
        plt.colorbar(plot[3],fraction=0.04, pad=0.04, aspect=10)
        ax.set_title(session[-16:-6], fontsize = 13)
        #ax.set_ylabel('Trials / Session', fontsize = 10)
        #ax.set_xlabel('Sessions', fontsize = 10)
    except Exception: 
        print (session + '/error')
        continue       

f4.tight_layout()
f4.subplots_adjust(top = 0.87)


#####SAVINGS#######


#main folder rat ID
script_dir = os.path.join(hardrive_path + 'Videogame_Assay/' + rat_ID)
#create a folder where to store the plots 
main_folder = os.path.join(script_dir +'/Summary')
#create a folder where to save the plots
results_dir = os.path.join(main_folder + '/Behaviour/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#save the fig in .tiff
f0.savefig(results_dir + figure_name0, transparent=True)
f1.savefig(results_dir + figure_name1, transparent=True)
f2.savefig(results_dir + figure_name2, transparent=True)
f3.savefig(results_dir + figure_name3, transparent=True)
f4.savefig(results_dir + figure_name4, transparent=True)
#f.savefig(results_dir + figure_name)      
    








