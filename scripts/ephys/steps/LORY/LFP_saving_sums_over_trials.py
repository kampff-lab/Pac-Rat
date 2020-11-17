# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:12:55 2020

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
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
# Reload modules
import importlib
importlib.reload(prs)
importlib.reload(behaviour)
importlib.reload(ephys)


#testing folder 
main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

hardrive_path = r'F:/'



#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']



#LEFT = BACK / RIGHT = FRONT / TOP = TOP /BOTTOM = BOTTOM)
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


#####FREQ BANDS OF INTEREST

#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz



RAT_ID = RAT_ID_ephys [1]
rat_summary_table_path=rat_summary_ephys[1]

probe_map_flatten = ephys.probe_map.flatten()


for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)

    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
        
        
        figure_folder = '/LFP/'
        
        results_dir =os.path.join(session_path + figure_folder)
        
        
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        #recording data path
        #raw_recording = os.path.join(session_path +'/Amplifier.bin')
        #downsampled_recording = os.path.join(session_path +'/Amplifier_downsampled.bin')
        #cleaned_recording =  os.path.join(session_path +'/Amplifier_cleaned.bin')
        
        #filed needed path 
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
        
        trial_end = event_finder(trial_end_idx, video_csv, samples_for_frames_file_path)
        touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
        ball = event_finder(ball_on, video_csv, samples_for_frames_file_path)
    
    
        downsampled_touch = np.uint32(np.array(touching_light)/30)
        downsampled_ball = np.uint32(np.array(ball)/30)
        downsampled_end= np.uint32(np.array(trial_end)/30)
        #end_samples = event_finder(trial_end_idx,video_csv,samples_for_frames_file_path)
        #samples_lenght_end_to_end = np.diff(np.hstack((0, end_samples)))
        #sample_start_clip = end_samples[21]
        #clip_sample_lenght = samples_lenght_end_to_end[22]


        data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
        down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
        num_samples = int(int(len(down))/num_raw_channels)
        reshaped_down=  np.reshape(down,(num_samples,num_raw_channels))  
        down=None
        down_T = reshaped_down.T

        freq = 30000
        offset = 1500
        num_raw_channels = 128
        

        #remove the first early trials
        downsampled_event_idx = downsampled_touch[1:]
        
        event_name= 'touch.csv'
         
        
    
        
        csv_alpha_pre = 'sum_alpha_before_ch_over_trials_' + event_name  #[r]
        
        csv_beta_pre ='sum_beta_before_ch_over_trials_' +event_name

        csv_delta_pre =  'sum_delta_before_ch_over_trials_' +event_name
   
        csv_theta_pre = 'sum_theta_before_ch_over_trials_' +event_name
        
        
        csv_alpha_post =  'sum_alpha_after_ch_over_trials_' +event_name
        
        csv_beta_post =   'sum_beta_after_ch_over_trials_' +event_name

        csv_delta_post =   'sum_delta_after_ch_over_trials_' +event_name
   
        csv_theta_post =  'sum_theta_after_ch_over_trials_' +event_name     

          
        delta_pre = np.zeros((N,len(downsampled_event_idx)))
        delta_post = np.zeros((N,len(downsampled_event_idx)))
        theta_pre = np.zeros((N,len(downsampled_event_idx)))
        theta_post = np.zeros((N,len(downsampled_event_idx)))        
        alpha_pre = np.zeros((N,len(downsampled_event_idx)))
        alpha_post = np.zeros((N,len(downsampled_event_idx)))        
        beta_pre = np.zeros((N,len(downsampled_event_idx)))
        beta_post = np.zeros((N,len(downsampled_event_idx)))


        
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten
            try:
                        
                
                ch_downsampled = down_T[channel,:]#should use channel
                #down_T=None
        
                chunk_before = np.zeros((len(downsampled_event_idx),offset))
                
                chunk_after = np.zeros((len(downsampled_event_idx),offset))
        
                for e, event in enumerate(downsampled_event_idx):
                     
                    chunk_before[e,:] = ch_downsampled[event-offset : event]
                    chunk_after[e,:] = ch_downsampled[event : event+offset]
                    
                    print(e)
                print('epoch_event_DONE')
        
           

                #half size chunk double  bandwidth   
                ch_downsampled = None
                
                    
                p_before, f_before = time_frequency.psd_array_multitaper(chunk_before, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
        
                p_after, f_after= time_frequency.psd_array_multitaper(chunk_after, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
        
                
                
                delta_trial_sum_pre = []               
                delta_trial_sum_post = []
                theta_trial_sum_pre = []               
                theta_trial_sum_post = []
                alpha_trial_sum_pre = []               
                alpha_trial_sum_post = []
                beta_trial_sum_pre = []               
                beta_trial_sum_post = []
                
                
                for trial in np.arange(len(downsampled_event_idx)):
                    
                    
                    
                    delta_ch = [i for i,v in enumerate(f_before) if 1< v <4 ]
                    delta_sel = p_before[:][trial][delta_ch]    
                    delta_sum = np.sum(delta_sel)
                    delta_trial_sum_pre.append(delta_sum)
                    
                    
                    delta_ch = [i for i,v in enumerate(f_after) if 1< v <4 ]
                    delta_sel = p_after[:][trial][delta_ch]            
                    delta_sum = np.sum(delta_sel)             
                    delta_trial_sum_post.append(delta_sum)   
                 
                    
                    
                    theta_ch = [i for i,v in enumerate(f_before) if 4< v <8 ]
                    theta_sel = p_before[:][trial][theta_ch]
                    theta_sum = np.sum(theta_sel)
                    theta_trial_sum_pre.append(theta_sum)
                    
                    theta_ch = [i for i,v in enumerate(f_after) if 4< v <8 ]
                    theta_sel = p_after[:][trial][theta_ch]              
                    theta_sum = np.sum(theta_sel)             
                    theta_trial_sum_post.append(theta_sum)
                        
                                    
                    alpha_ch = [i for i,v in enumerate(f_before) if 8< v <12 ]
                    alpha_sel = p_before[:][trial][alpha_ch]
                    alpha_sum = np.sum(alpha_sel)           
                    alpha_trial_sum_pre.append(alpha_sum)
                    
                    alpha_ch = [i for i,v in enumerate(f_after) if 8< v <12 ]
                    alpha_sel = p_after[:][trial][alpha_ch]
                    alpha_sum = np.sum(alpha_sel)               
                    alpha_trial_sum_post.append(alpha_sum)                 
                    
                                       
                    beta_ch = [i for i,v in enumerate(f_before) if 12< v <30 ]
                    beta_sel = p_before[:][trial][beta_ch]             
                    beta_sum= np.sum(beta_sel)             
                    beta_trial_sum_pre.append(beta_sum)

                    
                    beta_ch = [i for i,v in enumerate(f_after) if 12< v <30 ]
                    beta_sel =  p_after[:][trial][beta_ch]               
                    beta_sum= np.sum(beta_sel)               
                    beta_trial_sum_post.append(beta_sum)
                    
                    
    

            except Exception:
                continue 
             
            alpha_pre[ch,:]= alpha_trial_sum_pre
            beta_pre[ch,:] = beta_trial_sum_pre
            delta_pre[ch,:] = delta_trial_sum_pre
            theta_pre[ch,:] = theta_trial_sum_pre
            
      
            alpha_post[ch,:]= alpha_trial_sum_post
            beta_post[ch,:] = beta_trial_sum_post
            delta_post[ch,:] = delta_trial_sum_post
            theta_post[ch,:] = beta_trial_sum_post
        

                                
        print('session_done')

        
        #saving pre event
        np.savetxt(results_dir +  csv_alpha_pre, alpha_pre,delimiter=',', fmt='%s')
        np.savetxt(results_dir +  csv_beta_pre, beta_pre,delimiter=',', fmt='%s')
        np.savetxt(results_dir  + csv_delta_pre, delta_pre,delimiter=',', fmt='%s')
        np.savetxt(results_dir +  csv_theta_pre, theta_pre,delimiter=',', fmt='%s')
        
        
    
        #saving post event 
        np.savetxt(results_dir +  csv_alpha_post, alpha_post,delimiter=',', fmt='%s')
        np.savetxt(results_dir +  csv_beta_post, beta_post,delimiter=',', fmt='%s')
        np.savetxt(results_dir +  csv_delta_post, delta_post,delimiter=',', fmt='%s')
        np.savetxt(results_dir +  csv_theta_post, theta_post,delimiter=',', fmt='%s')
    



#################################################################
        
#retrieve saved files and create 11x11 hist to check distribution pre VS post and t test for each ch 
#files are saved with words after and before so alfabetically after is [0] and before is [1] in the matching array         
band = 'beta'
       

        
for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)

    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)    
        csv_dir_path = os.path.join(session_path + figure_folder)
       
        matching_files_before  = np.array(glob.glob(csv_dir_path +"*"+band+"*" + "*before*" ))
        sum_before = np.genfromtxt(matching_files_before[0], delimiter= ',',dtype= float)       
        
        matching_files_after = np.array(glob.glob(csv_dir_path +"*"+band+"*" + "*after*" ))
        sum_after = np.genfromtxt(matching_files_after[0], delimiter= ',',dtype= float)
 
        f0 =plt.figure(figsize=(20,20))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine() 
                 
        probe_t_test= []
        
        
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten #probe_map_flatten
           
              
            
            ch_t_test = stats.ttest_rel(sum_before[ch,:],sum_after[ch,:])
            probe_t_test.append(ch_t_test[1])

           
            ax = f0.add_subplot(11, 11, 1+ch, frameon=False)
            
            plt.hist(sum_before[ch,:],color= 'green', bins=20, alpha=.5,  linewidth=1,label='PRE touch')            
            plt.hist(sum_after[ch,:], color='red' ,bins=20,alpha=.5,  linewidth=1, label='POST touch')
            
               
            
        plt.suptitle(session)
        plt.legend(loc='upper right')  
        plt.close()           


    f1 =plt.figure(figsize=(10,10))
    sns.set()
    sns.set_style('white')
    sns.axes_style('white')
    sns.despine() 
    
    
    t_test_heatmap =np.reshape(probe_t_test,newshape=probe_map.shape) 
         
    ax = sns.heatmap(t_test_heatmap,annot=True,  cmap="bwr",vmin = 0, vmax=1,  edgecolors='white', linewidths=1,
                                  annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm() # "YlGnBu" RdBu
    
    #ax.patch.set(hatch='//', edgecolor='black')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.suptitle(session)
    
    from statsmodels.sandbox.stats.multicomp import multipletests
    p_adjusted = multipletests(probe_t_test,alpha=.5, method='bonferroni') # 0.05/121 = 0.0004132231404958678







