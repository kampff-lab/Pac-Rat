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

rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 
rat_ID = 'AK_33.2'


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
    

##########################################################################################

hardrive_path = r'F:/' 

#snippets idx:  trial start = 0  / trial end = 1 / ball touch = 2

def create_tracking_snippets_start_to_touch_and_touch_to_end(sessions_subset,start_snippet_idx=0,end_snippet_idx=1,mid_snippet_idx=2):
    
    x = len(sessions_subset)
    
    for count in np.arange(x):
        try:
        
            session = sessions_subset[count]                
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(hardrive_path, session + '/events/')
            trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv')
            ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')
            
            
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)    
            ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float)
            
            x_nan_nose, y_nan_nose = DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3)
            x_nan_tail_base, y_nan_tail_base = DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 10, dlc_y_column = 11, dlc_likelihood_column = 12)
    
        
            l = len(ball_coordinates)
            x_nose_trial_tracking_snippets_start_to_touch = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_start_to_touch = [[] for _ in range(l)] 
            x_tail_base_trial_tracking_start_to_touch = [[] for _ in range(l)]
            y_tail_base_trial_tracking_start_to_touch = [[] for _ in range(l)] 
       

            trial_lenght_start_to_touch = abs(trial_idx[:,start_snippet_idx] - trial_idx[:,mid_snippet_idx])
            start_idx = trial_idx[:,start_snippet_idx]
                               
            count_1 = 0
            
            for start in start_idx:
                x_nose_trial_tracking_snippets_start_to_touch[count_1] = x_nan_nose[start:start + trial_lenght_start_to_touch[count_1]]
                y_nose_trial_tracking_snippets_start_to_touch[count_1] = y_nan_nose[start:start + trial_lenght_start_to_touch[count_1]]
                x_tail_base_trial_tracking_start_to_touch[count_1] = x_nan_tail_base[start:start + trial_lenght_start_to_touch[count_1]]
                y_tail_base_trial_tracking_start_to_touch[count_1] = y_nan_tail_base[start:start + trial_lenght_start_to_touch[count_1]]
                count_1 += 1
            
            
            trial_lenght_touch_to_end = abs(trial_idx[:,mid_snippet_idx] - trial_idx[:,end_snippet_idx])
            touch_idx = trial_idx[:,mid_snippet_idx]        
            
    
            x_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            x_tail_base_trial_tracking_touch_to_end = [[] for _ in range(l)]
            y_tail_base_trial_tracking_touch_to_end = [[] for _ in range(l)] 
    
    
            
            count_2 = 0
            
            for touch in touch_idx:
                x_nose_trial_tracking_snippets_touch_to_end[count_2] = x_nan_nose[touch:touch + trial_lenght_touch_to_end[count_2]]
                y_nose_trial_tracking_snippets_touch_to_end[count_2] = y_nan_nose[touch:touch + trial_lenght_touch_to_end[count_2]]
                x_tail_base_trial_tracking_touch_to_end[count_2] = x_nan_tail_base[touch:touch + trial_lenght_touch_to_end[count_2]]
                y_tail_base_trial_tracking_touch_to_end[count_2] = y_nan_tail_base[touch:touch + trial_lenght_touch_to_end[count_2]]
                count_2 += 1
                   
            
            
            f=plt.figure(figsize=(20,10))
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine()
            for row in np.arange(l):
                plt.plot(x_nose_trial_tracking_snippets_start_to_touch[row],y_nose_trial_tracking_snippets_start_to_touch[row], '.',color ='#FF7F50', alpha=.1)
                plt.plot(x_nose_trial_tracking_snippets_touch_to_end[row],y_nose_trial_tracking_snippets_touch_to_end[row],'.',color = '#ADFF2F', alpha=.1)
                plt.plot(ball_coordinates[row][0],ball_coordinates[row][1],'o', color ='k') 
                plt.title('Nose_trial_tracking'+ '_session%d' %count, fontsize = 16)
            f.tight_layout()
            f.savefig('F:/test_folder/nose_level_2/'+'_session%d' %count)
                
    
    
            f1=plt.figure(figsize=(20,10))
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine()
            for row in np.arange(l):
                plt.plot(x_tail_base_trial_tracking_start_to_touch[row],y_tail_base_trial_tracking_start_to_touch[row], '.',color = '#FF7F50', alpha=.1)
                plt.plot(x_tail_base_trial_tracking_touch_to_end[row],y_tail_base_trial_tracking_touch_to_end[row],'.',color = '#ADFF2F', alpha=.1)
                plt.plot(ball_coordinates[row][0],ball_coordinates[row][1],'o', color ='k') 
                plt.title('Tail_base_trial_tracking'+ '_session%d' %count, fontsize = 16)
            f1.tight_layout()
            f1.savefig('F:/test_folder/tail_level_2/'+'_session%d' %count)
            
            
            print(count)        
        
        except Exception: 
            print (session + '/error')
            continue   
   





def create_tracking_snippets_start_to_end_trial(sessions_subset,start_snippet_idx=0,end_snippet_idx=1):
    
    x = len(sessions_subset)
    
    for count in np.arange(x):
        try:
        
            session = sessions_subset[count]                
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(hardrive_path, session + '/events/')
            trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv')

            
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)    
            
            x_nan_nose, y_nan_nose = DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3)
            x_nan_tail_base, y_nan_tail_base = DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 10, dlc_y_column = 11, dlc_likelihood_column = 12)
    
        
            l = len(trial_idx)
            x_nose_trial_tracking_snippets_start_to_end = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_start_to_end = [[] for _ in range(l)] 
            x_tail_base_trial_tracking_start_to_end = [[] for _ in range(l)]
            y_tail_base_trial_tracking_start_to_end = [[] for _ in range(l)] 
       

            trial_lenght_start_to_end = abs(trial_idx[:,start_snippet_idx] - trial_idx[:,end_snippet_idx])
            start_idx = trial_idx[:,start_snippet_idx]
                               
            count_1 = 0
            
            for start in start_idx:
                x_nose_trial_tracking_snippets_start_to_end[count_1] = x_nan_nose[start:start + trial_lenght_start_to_end[count_1]]
                y_nose_trial_tracking_snippets_start_to_end[count_1] = y_nan_nose[start:start + trial_lenght_start_to_end[count_1]]
                x_tail_base_trial_tracking_start_to_end[count_1] = x_nan_tail_base[start:start + trial_lenght_start_to_end[count_1]]
                y_tail_base_trial_tracking_start_to_end[count_1] = y_nan_tail_base[start:start + trial_lenght_start_to_end[count_1]]
                count_1 += 1
            
            
           
            f=plt.figure(figsize=(20,10))
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine()
            for row in np.arange(l):
                plt.plot(x_nose_trial_tracking_snippets_start_to_end[row],y_nose_trial_tracking_snippets_start_to_end[row], '.',color ='#FF7F50', alpha=.1)
                plt.title('Nose_trial_tracking'+ '_session%d' %count, fontsize = 16)
            f.tight_layout()
            f.savefig('F:/test_folder/nose_level_1/'+'_session%d' %count)
                
    
    
            f1=plt.figure(figsize=(20,10))
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine()
            for row in np.arange(l):
                plt.plot(x_tail_base_trial_tracking_start_to_end[row],y_tail_base_trial_tracking_start_to_end[row], '.',color = '#1E90FF', alpha=.1)
                plt.title('Tail_base_trial_tracking'+ '_session%d' %count, fontsize = 16)
            f1.tight_layout()
            f1.savefig('F:/test_folder/tail_level_1/'+'_session%d' %count)
            
            
            print(count)        
        
        except Exception: 
            print (session + '/error')
            continue   
   








def create_colorcoded_snippets_start_to_touch_and_touch_to_end(sessions_subset,start_snippet_idx=0,end_snippet_idx=1,mid_snippet_idx=2):
    
    x = len(sessions_subset)
    
    for count in np.arange(x):
        try:
        
            session = sessions_subset[count]                
            script_dir = os.path.join(hardrive_path + session) 
            csv_dir_path = os.path.join(hardrive_path, session + '/events/')
            trial_idx_path = os.path.join(hardrive_path, session + '/events/' + 'Trial_idx.csv')
            ball_coordinates_path = os.path.join(hardrive_path, session + '/events/' + 'Ball_coordinates.csv')
            
            
            trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)    
            ball_coordinates = np.genfromtxt(ball_coordinates_path, delimiter = ',', dtype = float)
            
            x_nan_nose, y_nan_nose = DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3)
            x_nan_tail_base, y_nan_tail_base = DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 10, dlc_y_column = 11, dlc_likelihood_column = 12)
    
        
            l = len(ball_coordinates)
            x_nose_trial_tracking_snippets_start_to_touch = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_start_to_touch = [[] for _ in range(l)] 
       
            trial_lenght_start_to_touch = abs(trial_idx[:,start_snippet_idx] - trial_idx[:,mid_snippet_idx])
            start_idx = trial_idx[:,start_snippet_idx]
                               
            count_1 = 0
            
            for start in start_idx:
                x_nose_trial_tracking_snippets_start_to_touch[count_1] = x_nan_nose[start:start + trial_lenght_start_to_touch[count_1]]
                y_nose_trial_tracking_snippets_start_to_touch[count_1] = y_nan_nose[start:start + trial_lenght_start_to_touch[count_1]]
                count_1 += 1
            
            
            trial_lenght_touch_to_end = abs(trial_idx[:,mid_snippet_idx] - trial_idx[:,end_snippet_idx])
            touch_idx = trial_idx[:,mid_snippet_idx]        
            
    
            x_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
            y_nose_trial_tracking_snippets_touch_to_end = [[] for _ in range(l)] 
   
            count_2 = 0
            
            for touch in touch_idx:
                x_nose_trial_tracking_snippets_touch_to_end[count_2] = x_nan_nose[touch:touch + trial_lenght_touch_to_end[count_2]]
                y_nose_trial_tracking_snippets_touch_to_end[count_2] = y_nan_nose[touch:touch + trial_lenght_touch_to_end[count_2]]

                count_2 += 1
            
            
            lenght=len(x_nose_trial_tracking_snippets_touch_to_end)   
            y_nose_meeting_criteria = [[] for _ in range(lenght)] 
            idx_meeting_criteria= []
            ball_coordinates_subset=[]
            x_nose_meeting_criteria = [[] for _ in range(lenght)]
            
          
            for i in np.arange(lenght):
                select_x = np.sort(x_nose_trial_tracking_snippets_touch_to_end[i])
                select_y = np.sort(y_nose_trial_tracking_snippets_touch_to_end[i])   
                good_x_indices = select_x >1100
                potential_y = select_y[good_x_indices]
                if potential_y.size ==0:
                    continue
                max_value_y =max(potential_y)
                if max_value_y>910: 
                    y_nose_meeting_criteria[i]=y_nose_trial_tracking_snippets_touch_to_end[i]
                    x_nose_meeting_criteria[i]=x_nose_trial_tracking_snippets_touch_to_end[i]
                    ball_coordinates_subset.append(ball_coordinates[i]) 
                    idx_meeting_criteria.append(i)
            
            
            trial_list = np.array(list(range(len(ball_coordinates))))

            left_out_trials=np.delete(trial_list,np.array(idx_meeting_criteria))
            left_out_x_nose=np.array(x_nose_trial_tracking_snippets_touch_to_end)[left_out_trials]
            left_out_y_nose=np.array(y_nose_trial_tracking_snippets_touch_to_end)[left_out_trials]
            
            l=len(idx_meeting_criteria)       
            f=plt.figure(figsize=(20,10)) 
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine()
            #colors = plt.cm.jet(np.linspace(0,1,l))
        
            for row in np.arange(l):  
                plt.plot(x_nose_meeting_criteria[row],y_nose_meeting_criteria[row],color = 'r', alpha=.5)
                #plt.plot(ball_coordinates_subset[row][0],ball_coordinates_subset[row][1],'o', color ='k')
                plt.plot(left_out_x_nose[row],left_out_y_nose[row],color = 'b', alpha=.5)
                #plt.plot(ball_coordinates_subset[row][0],ball_coordinates_subset[row][1],'o', color ='k') 
                plt.title('Nose_subset'+ '_session%d' %count, fontsize = 16)
            f.tight_layout()
            f.savefig('F:/test_folder/nose_level_2/'+'_session%d' %count)

            print(count)        
        
        except Exception: 
            print (session + '/error')
            continue
 
    
    
    
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
test=np.array(x_nose_trial_tracking_snippets_touch_to_end)

missing_element = []
for i in range(idx_meeting_criteria[0], idx_meeting_criteria[-1]+1):
    if i not in idx_meeting_criteria:
        missing_element.append(i)

print( missing_element)

trial_list=np.array(list(range(len(ball_coordinates))))

left_trial=np.delete(trial_list,criteria_array)

criteria_array=np.array(idx_meeting_criteria)
test2=[(e1+1) for e1,e2 in zip(idx_meeting_criteria, idx_meeting_criteria[1:]) if e2-e1 != 1]
test2=test[idx_meeting_criteria]
y_nose_meeting_criteria = [[] for _ in range(lenght)] 
idx_meeting_criteria= []
ball_coordinates_subset=[]
x_nose_meeting_criteria = [[] for _ in range(lenght)]

lenght=len(x_nose_trial_tracking_snippets_touch_to_end)
#row= x_nose_trial_tracking_snippets_touch_to_end[i]>1200           
 


for i in np.arange(lenght):
    select_x = np.sort(x_nose_trial_tracking_snippets_touch_to_end[i])
    select_y = np.sort(y_nose_trial_tracking_snippets_touch_to_end[i])    
    max_value_x = max(select_x)
    max_value_y = max(select_y)
    if max_value_y>910 and max_value_x>1200: 
        y_nose_meeting_criteria[i]=y_nose_trial_tracking_snippets_touch_to_end[i]
        x_nose_meeting_criteria[i]=x_nose_trial_tracking_snippets_touch_to_end[i]
        ball_coordinates_subset.append(ball_position[i]) 
        idx_meeting_criteria.append(i)


test=x_nose_trial_tracking_snippets_touch_to_end[]


        
i=0

for i in np.arange(lenght):
   y_nose= y_nose_trial_tracking_snippets_touch_to_end[i]
   x_nose =x_nose_trial_tracking_snippets_touch_to_end[i]    
    for trial in 
    if max_value_y>910 and max_value_x>1200: 
        y_nose_meeting_criteria[i]=y_nose_trial_tracking_snippets_touch_to_end[i]
        x_nose_meeting_criteria[i]=x_nose_trial_tracking_snippets_touch_to_end[i]
        ball_coordinates_subset.append(ball_position[i]) 
        idx_meeting_criteria.append(i)





 
l=len(idx_meeting_criteria)       
plt.figure(figsize=(20,10)) 


colors = plt.cm.jet(np.linspace(0,1,l))
#c=next(color)
for row in np.arange(l):  
    plt.plot(x_nose_meeting_criteria[row],y_nose_meeting_criteria[row],color = colors[row], alpha=.5)
    plt.plot(ball_coordinates_subset[row][0],ball_coordinates_subset[row][1],'o', color ='k') 
    
      
        
test2 = np.array(y_nose_trial_tracking_snippets_touch_to_end)
test3 =test2[idx][0]



plt.figure(figsize=(20,10))  
plt.plot(x_nose_trial_tracking_snippets_touch_to_end[i],y_nose_trial_tracking_snippets_touch_to_end[i],color = 'r', alpha=.5)




















#find idx of the balls accordng to the quadrants in which they appear

      
quadrant_1,quadrant_2,quadrant_3,quadrant_4=ball_positions_based_on_quadrant_of_appearance(session)


    
            
plt.plot(ball_position[quadrant_1][:,0],ball_position[quadrant_1][:,1],'o',color='r')
plt.plot(ball_position[quadrant_2][:,0],ball_position[quadrant_2][:,1],'o',color='g')
plt.plot(ball_position[quadrant_3][:,0],ball_position[quadrant_3][:,1],'o',color='b')
plt.plot(ball_position[quadrant_4][:,0],ball_position[quadrant_4][:,1],'o',color='m')

                
                
    
plt.figure(figsize=(20,10))
for row in quadrant_1:
    plt.plot(x_nose_trial_tracking_snippets_touch_to_end[row],y_nose_trial_tracking_snippets_touch_to_end[row],color = 'r', alpha=.5)
plt.plot(ball_position[quadrant_1][:,0],ball_position[quadrant_1][:,1],'o',color='r')
plt.figure(figsize=(20,10))
for row in quadrant_2:
    plt.plot(x_nose_trial_tracking_snippets_touch_to_end[row],y_nose_trial_tracking_snippets_touch_to_end[row],color = 'g', alpha=.5)
plt.plot(ball_position[quadrant_2][:,0],ball_position[quadrant_2][:,1],'o',color='g')
plt.figure(figsize=(20,10))
for row in quadrant_3:
    plt.plot(x_nose_trial_tracking_snippets_touch_to_end[row],y_nose_trial_tracking_snippets_touch_to_end[row],color = 'b', alpha=.5)
plt.plot(ball_position[quadrant_3][:,0],ball_position[quadrant_3][:,1],'o',color='b')
plt.figure(figsize=(20,10))  
for row in quadrant_4:
    plt.plot(x_nose_trial_tracking_snippets_touch_to_end[row],y_nose_trial_tracking_snippets_touch_to_end[row],color = 'm', alpha=.5)
plt.plot(ball_position[quadrant_4][:,0],ball_position[quadrant_4][:,1],'o',color='m')#create plot centered to ball




#x_nose_centered = [[] for _ in range(l)]
#y_nose_centered = [[] for _ in range(l)]
#x_tail_base_centered = [[] for _ in range(l)]
#y_tail_base_centered = [[] for _ in range(l)]
#    
#for count in np.arange(l):
#
#
#    x_nose_centered[count] = x_nose_trial_tracking_start_to_end[count]-ball_coordinates[count][0]
#    y_nose_centered[count] = y_nose_trial_tracking_start_to_end[count]-ball_coordinates[count][1]
#    x_tail_base_centered[count] = x_tail_base_trial_tracking_start_to_end[count]-ball_coordinates[count][0]
#    y_tail_base_centered[count] = y_tail_base_trial_tracking_start_to_end[count]-ball_coordinates[count][1]
#    



#
#for count in np.arange(l):
#    
#    plt.plot(x_nose_centered[count],y_nose_centered[count])
#
#
#
#
#for count in np.arange(l):
#    
#    plt.plot(x_tail_base_centered[count],y_tail_base_centered[count])
#    plt.plot(0,0,'o',color='k')
#    
    
    
    
    