# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:09:32 2021

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
import mne
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from scipy import stats 
from mne import time_frequency
import parser_library as prs
import behaviour_library as behaviour
import ephys_library as ephys 
#import plotting_probe_layout as layout
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import glob
from statsmodels.sandbox.stats.multicomp import multipletests
# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)

#test George cluster code 
hardrive_path = r'F:/'



#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']


RAT_ID = RAT_ID_ephys

rat_summary_table_path=rat_summary_ephys

#LEFT = BACK / RIGHT = FRONT / TOP = TOP /BOTTOM = BOTTOM)
probe_map = np.array([[103,78,81,118,94,74,62,24,49,46,7],
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




probe_map_flatten = ephys.probe_map.flatten()



RAT_ID = RAT_ID_ephys[0]

rat_summary_table_path=rat_summary_ephys[0]


#REWARDED TRIALS  (NO TRIALS EXCLUDED)

for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    offset=3000#1500
 
   
    events= ['touch','ball','reward']
    
    for ev in np.arange(len(events)):
       
        for s, session in enumerate(sessions_subset): 
            
    
                  
            session_path =  os.path.join(hardrive_path,session)
            success,misses = behaviour.trial_outcome_index(session_path)
            
            #files needed path 
            csv_dir_path = os.path.join(session_path + '/events/')
            touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
            ball_on = os.path.join(hardrive_path, session +'/events/'+'BallON.csv')
            #trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
            trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
            #trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)       
            video_csv = os.path.join(session_path + '/Video.csv')       
            samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
            
            
            #files opening
            samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)
            
            
            #find eveNt of interest
            trial_end = event_finder(trial_end_idx, video_csv, samples_for_frames_file_path)
            touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
            ball = event_finder(ball_on, video_csv, samples_for_frames_file_path)
        
            #downsampling of the event sample
            downsampled_touch = np.uint32(np.array(touching_light)/30)
            downsampled_ball = np.uint32(np.array(ball)/30)
            downsampled_end= np.uint32(np.array(trial_end)/30)
    
            events_list = [downsampled_touch, downsampled_ball, downsampled_end]
    
            #pick event of interest and exclude first trials because it is too noisy
            downsampled_event_idx = events_list[ev]#[1:]
            
            #electe rewarded and event trials based on the idx, substract 1 because the first trial has been removed 
            rewarded_trials = downsampled_event_idx[np.array(success)]# -1]
           
            #trial 1 which has idx =0 becomes -1 therefore the last idx is chosen 
            #given we esclude the first trial, when the first idx is greater than the second remove it 
    
    #        if missed
    #
    #
    #        if missed_trials[0]>missed_trials[1]:
    #            
    #            missed_trials=missed_trials[1:]
    #        else:
    #            
    #            rewarded_trials=rewarded_trials[1:]
                
    
            print(len(downsampled_event_idx))
           
    
            #opening binned file
            binned_mua_path = os.path.join(session_path +'/Amplifier_cleaned__BINNED.bin')
            binned_mua_raw = np.fromfile(binned_mua_path, dtype=np.uint8)        
            binned_mua_reshape =  np.reshape(binned_mua_raw,(128,-1))
            #mua_mapped = binned_mua_reshape[probe_map_flatten]
    
    
    
            reward_matrix_avg = np.zeros((121,int(offset*2/10),len(rewarded_trials)))
            
            reward_matrix_std = np.zeros((121,int(offset*2/10),len(rewarded_trials)))
            
            
            #finding chunks 
            for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten
                            
                #select one ch     
                ch_bin = binned_mua_reshape[channel,:]
               
                #chunk_reward = np.zeros((len(rewarded_trials),int(offset*2/10)))
     
                bin_size = 10   
                ch_sum =  [sum(ch_bin[i:i+bin_size]) for i in range(0, len(ch_bin)-10, bin_size)]
                ch_avg = np.mean(ch_sum) # baseline spiking
                ch_std = np.std(ch_sum)   
                    
                 
                for e, event in enumerate(rewarded_trials):
                    
                    #chunk around trial 
                    reward_snippet = ch_bin[event-offset : event+offset]
                    
                    sum_reward = [sum(reward_snippet[i:i+bin_size]) for i in range(0, len(reward_snippet), bin_size)]
                   
                    
                    R_avg = (sum_reward-ch_avg)/ch_avg
                    R_std = (sum_reward-ch_avg)/ch_std 
                    
                    reward_matrix_avg[ch,:,e]= R_avg
                    reward_matrix_std[ch,:,e]= R_std  
                    
                print(ch,channel)   
                    
                    #print(e)
                    
              
                
            avg_reward_matrix_avg = np.mean(reward_matrix_avg, axis=-1) 
            avg_reward_matrix_std = np.mean(reward_matrix_std, axis=-1)
            #print(np.shape(avg_reward_matrix))   
        
            #plot 11x11 probe layout  mean      
            f =plt.figure(figsize=(20,20))
        
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine() 
                   
            for c  in np.arange(len(probe_map_flatten)):
                   
                    
                ax = f.add_subplot(11, 11, 1+c, frameon=False)
    
                
                plt.plot(avg_reward_matrix_avg[c])
                plt.vlines(int(offset/10), 0,max(avg_reward_matrix_avg[c]), 'r')
                
                plt.ylim(-1,30)
                plt.xticks(fontsize=5, rotation=90)
                plt.yticks(fontsize=5, rotation=0)
                #plt.xticks(np.arange(0, 30))
                #plt.yticks(np.arange(0, 300))
                
                plt.suptitle( RAT_ID[r] + 'session_'+str(s)+ 'rewarded_' + events[ev])
                
                
            plt.tight_layout()
            figure_name = RAT_ID[r] + '_session_'+str(s)+ '_rewarded_'+ events[ev]+'_psth.png'
            
            f.savefig('F:/Videogame_Assay/MUA_summary_plots/'+figure_name)
            plt.close() 


            #plot 11x11 probe layout  std      
            f =plt.figure(figsize=(20,20))
        
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine() 
                   
            for c  in np.arange(len(probe_map_flatten)):
                   
                    
                ax = f.add_subplot(11, 11, 1+c, frameon=False)
    
               
                plt.plot(avg_reward_matrix_std[c])
                plt.vlines(int(offset/10), 0,max(avg_reward_matrix_std[c]), 'r')
                
                plt.ylim(-1,5)
                plt.xticks(fontsize=5, rotation=90)
                plt.yticks(fontsize=5, rotation=0)
                #plt.xticks(np.arange(0, 30))
                #plt.yticks(np.arange(0, 300))
                
                plt.suptitle( RAT_ID[r] + 'session_'+str(s)+ 'rewarded_ball')
                
                
            plt.tight_layout()
            figure_name = RAT_ID[r] + '_session_'+str(s)+ '_rewarded_ball_psth.png'
            f.savefig('F:/Videogame_Assay/MUA_summary_plots/'+figure_name)
            plt.close() 

        print(session)
    print(events[ev])
    














###################################################################################################

#misses
for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    offset=3000#1500
  
    tot_sessions = len(sessions_subset)
      
       
    for s, session in enumerate(sessions_subset): 
        
              
        session_path =  os.path.join(hardrive_path,session)
        success,misses = behaviour.trial_outcome_index(session_path)
        
        if misses==[]:
            break 
            
        #files needed path 
        csv_dir_path = os.path.join(session_path + '/events/')
        touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
        ball_on = os.path.join(hardrive_path, session +'/events/'+'BallON.csv')
        #trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
        trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
        #trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)       
        video_csv = os.path.join(session_path + '/Video.csv')       
        samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
        
        
        #files opening
        samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)
        
        
        #find evet of interest
        trial_end = event_finder(trial_end_idx, video_csv, samples_for_frames_file_path)
        touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
        ball = event_finder(ball_on, video_csv, samples_for_frames_file_path)
    
        #downsampling of the event sample
        downsampled_touch = np.uint32(np.array(touching_light)/30)
        downsampled_ball = np.uint32(np.array(ball)/30)
        downsampled_end= np.uint32(np.array(trial_end)/30)


        #exclude first trials because it is too noisy
        downsampled_event_idx = downsampled_ball#[1:]
        
        #electe rewarded and event trials based on the idx, substract 1 because the first trial has been removed 
      
        missed_trials= downsampled_event_idx[np.array(misses)]# -1]
        
        #trial 1 which has idx =0 becomes -1 therefore the last idx is chosen 
        #given we esclude the first trial, when the first idx is greater han the second remove it 

#        if missed
#
#
#        if missed_trials[0]>missed_trials[1]:
#            
#            missed_trials=missed_trials[1:]
#        else:
#            
#            rewarded_trials=rewarded_trials[1:]
            

        print(len(downsampled_event_idx))


        #open binned file
        binned_mua_path = os.path.join(session_path +'/Amplifier_cleaned__BINNED.bin')
        binned_mua_raw = np.fromfile(binned_mua_path, dtype=uint8)        
        binned_mua_reshape =  np.reshape(binned_mua_raw,(128,-1))
        #mua_mapped = binned_mua_reshape[probe_map_flatten]
        

        miss_matrix_avg = np.zeros((121,int(offset*2/10),len(missed_trials)))
        
        miss_matrix_std = np.zeros((121,int(offset*2/10),len(missed_trials)))
        

        
        #finding chunks 
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten
                        
             #select one channel    
            ch_bin = binned_mua_reshape[channel,:]
            
            
            bin_size = 10   
            #sum spikes in each 10ms bin
            ch_sum =  [sum(ch_bin[i:i+bin_size]) for i in range(0, len(ch_bin)-10, bin_size)]
            #means and std of the ch over the session
            ch_avg = np.mean(ch_sum)
            ch_std = np.std(ch_sum)   
       
            #chunk_miss =  np.zeros((len(missed_trials),int(offset*2/10)))
    

            for e, event in enumerate(missed_trials):
                        
                miss_snippet = ch_bin[event-offset : event+offset]
                
                sum_miss = [sum(miss_snippet[i:i+bin_size]) for i in range(0, len(miss_snippet), bin_size)]
             
                R_avg = (sum_miss-ch_avg)/ch_avg
                R_std = (sum_miss-ch_avg)/ch_std
                         
                
                
                miss_matrix_avg[ch,:,e]= R_avg
                miss_matrix_std[ch,:,e]= R_std
            #print(ch,channel)
          

        avg_miss_matrix = np.mean(miss_matrix, axis=-1)
        print(np.shape(avg_miss_matrix))
            



        
        f1 =plt.figure(figsize=(20,20))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine() 
        #outer_grid = gridspec.GridSpec(11, 11, wspace=0.0, hspace=0.0)
        
        for c  in np.arange(len(probe_map_flatten)):

                
                
            ax = f1.add_subplot(11, 11, 1+c, frameon=False)
            

            plt.vlines(150, 0,2, 'r')
            plt.plot(avg_miss_matrix[c])
            #plt.ylim(-.2,2)
            plt.xticks(fontsize=5, rotation=90)
            plt.yticks(fontsize=5, rotation=0) 
            plt.suptitle( RAT_ID[r] + 'session_'+str(s)+ '_missed_ball_psth')
            
        figure_name = RAT_ID[r] + '_session_'+str(s)+ '_missed_ball_psth.png'
        f1.savefig('F:/Videogame_Assay/MUA_summary_plots/'+figure_name)
        plt.close() 
        print(rat,r)

#
#        f1 =plt.figure(figsize=(20,20))
#        sns.set()
#        sns.set_style('white')
#        sns.axes_style('white')
#        sns.despine() 
#        #outer_grid = gridspec.GridSpec(11, 11, wspace=0.0, hspace=0.0)
#        #plt.figure()
#        for c  in np.arange(len(probe_map_flatten)): 
#            
#            ax = f1.add_subplot(11, 11, 1+c, frameon=False)
#            ch_sel = avg_miss_matrix[c]
#            for t, trial in enumerate(ch_sel):               
#                plt.vlines(ch_sel, t, t+1, color = [0,0,0,0.1])    
#           
#            print(c)
            
            
#            
#                plt.vlines(150, 0,2, 'r')
#                plt.plot(avg_miss_matrix[c,:,t])
#                #plt.ylim(-.2,2)
#                plt.xticks(fontsize=5, rotation=90)
#                plt.yticks(fontsize=5, rotation=0) 
#                plt.suptitle( RAT_ID[r] + 'session_'+str(s)+ '_missed_reward_psth')
#                
#        figure_name = RAT_ID[r] + '_session_'+str(s)+ '_missed_reward_psth.png'
#        f1.savefig('F:/Videogame_Assay/MUA_summary_plots/'+figure_name)
#        plt.close() 
#        print(rat,r)
#          
#        
#        
        
        
        




#        f1 =plt.figure(figsize=(20,20))
#        sns.set()
#        sns.set_style('white')
#        sns.axes_style('white')
#        sns.despine() 
#        #outer_grid = gridspec.GridSpec(11, 11, wspace=0.0, hspace=0.0)
#        plt.figure()
#        for c  in np.arange(len(probe_map_flatten)): 
#            
#            ax = f1.add_subplot(11, 11, 1+c, frameon=False)
#            ch_sel = reward_matrix[c].T
#            for t, trial in enumerate(ch_sel):               
#                plt.vlines(trial, t, t+1, color = [0,0,0,0.1])    
#                #print(t)
#            print(c)     
##           
#            
#            for c  in np.arange(len(probe_map_flatten)):
#
#
#            
#                f =plt.figure()
#                sns.set()
#                sns.set_style('white')
#                sns.axes_style('white')
#                sns.despine() 
#                plt.vlines(150, 0,1, 'r')
#                plt.plot(avg_miss_matrix[c])
#                plt.ylim(-.2,1)
#                plt.xticks(fontsize=5, rotation=90)
#                plt.yticks(fontsize=5, rotation=0) 
#                figure_name = str(probe_map_flatten[c]) + 'reward_psth.png'
#                f.savefig('C:/Users/KAMPFF-LAB-ANALYSIS3/Desktop/test/'+figure_name)
#                plt.close()
#            
#                
 