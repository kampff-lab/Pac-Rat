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



RAT_ID = RAT_ID_ephys[0] #[0]
rat_summary_table_path=rat_summary_ephys[0]#[0]

probe_map_flatten = ephys.probe_map.flatten()
len(probe_map_flatten)

for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)

    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
        
        
        #figure_folder = '/LFP/'
        
        #results_dir =os.path.join(session_path + figure_folder)
        
        
#        if not os.path.isdir(results_dir):
#            os.makedirs(results_dir)
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
        
        #baseline_idx = np.arange(120000,num_samples-120000,6000) 
        #remove the first early trials
        downsampled_event_idx = downsampled_touch[1:]
        
        event_name= 'touch.csv'
         
        
        csv_alpha_b = RAT_ID[r]+ '_sum_of_avg_alpha_base.csv'
        
        csv_beta_b = RAT_ID[r] + '_sum_of_avg_beta_base.csv'

        csv_delta_b = RAT_ID[r] + '_sum_of_avg_delta_base.csv'
   
        csv_theta_b = RAT_ID[r]+ '_sum_of_avg_theta_base.csv' 
        
        
        csv_alpha_pre = RAT_ID[r]+'sum_alpha_before_ch_over_trials_' + event_name  #[r]
        
        csv_beta_pre =RAT_ID[r]+'sum_beta_before_ch_over_trials_' +event_name

        csv_delta_pre = RAT_ID[r]+ 'sum_delta_before_ch_over_trials_' +event_name
   
        csv_theta_pre = RAT_ID[r]+'sum_theta_before_ch_over_trials_' +event_name
        
        
        csv_alpha_post =RAT_ID[r]+  'sum_alpha_after_ch_over_trials_' +event_name
        
        csv_beta_post = RAT_ID[r]+  'sum_beta_after_ch_over_trials_' +event_name

        csv_delta_post = RAT_ID[r]+  'sum_delta_after_ch_over_trials_' +event_name
   
        csv_theta_post =RAT_ID[r]+  'sum_theta_after_ch_over_trials_' +event_name     

          
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
                    
#                 #baseline_chunk_around_event = ch_downsampled[baseline_idx-offset : baseline_idx+offset]
#                for b, base in enumerate(baseline_idx):
#                     
#                    baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
#                    #print(b)    
#                print('epoch_baseline_DONE')
#                print('epoch_event_DONE')
#        
#           

                #half size chunk double  bandwidth   
                ch_downsampled = None
                
                    
                p_before, f_before = time_frequency.psd_array_multitaper(chunk_before, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
        
                p_after, f_after= time_frequency.psd_array_multitaper(chunk_after, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
        
                #p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
        
                #p_base_avg = np.mean(p_base, axis =0) #[:len(downsampled_event_idx)]
                #p_base_sem = stats.sem(p_base, axis = 0)
                
                # tot baseline excluding noise and sum 
                
#                baseline_tot = [i for i,v in enumerate(f_base) if  v <45 or v>55 ]
#                baseline_sel = p_base_avg[baseline_tot]
#                baseline_sum_tot = np.sum(baseline_sel)
#                tot_base_sum.append(baseline_sum_tot)    
#                
                
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
band = 'delta'
       

        
for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)

    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)    
        csv_dir_path = os.path.join(session_path + figure_folder)
       
        matching_files_before  = np.array(glob.glob(csv_dir_path +"*"+band+"*" + "*before*" + "*"+event_name+"*"))
        sum_before = np.genfromtxt(matching_files_before[0], delimiter= ',',dtype= float)       
        
        matching_files_after = np.array(glob.glob(csv_dir_path +"*"+band+"*" + "*after*"+"*"+event_name+"*" ))
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
            
               
            
        plt.suptitle(session + band + event_name[:-4])
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
    plt.suptitle(session+band + event_name[:-4])
    
    from statsmodels.sandbox.stats.multicomp import multipletests
    p_adjusted = multipletests(probe_t_test,alpha=.5, method='bonferroni') # 0.05/121 = 0.0004132231404958678




#########################create 1 file for each rat with all the trials
    

band = ['delta','theta','beta','alpha']
event_name = 'ball_on'   
#RAT_ID=RAT_ID[0]


for b in range(len(band)):

            
    for r, rat in enumerate(rat_summary_table_path): 
        
        
        #rat = rat_summary_table_path
        Level_2_post = prs.Level_2_post_paths(rat)
        sessions_subset = Level_2_post
        
        N = 121
        tot_sessions = len(sessions_subset)
        
        tot_trial_before = [[] for _ in range(tot_sessions)]
        tot_trial_after = [[] for _ in range(tot_sessions)]
        
        
        for s, session in enumerate(sessions_subset):        
           
            
            session_path =  os.path.join(hardrive_path,session)    
            csv_dir_path = os.path.join(session_path + figure_folder)
           
            matching_files_before  = np.array(glob.glob(csv_dir_path +"*"+band[b]+"*" + "*before*"+"*"+event_name+"*"))
            sum_before = np.genfromtxt(matching_files_before[0], delimiter= ',',dtype= float)       
            
            matching_files_after = np.array(glob.glob(csv_dir_path +"*"+band[b]+"*" + "*after*" + "*"+event_name+"*"))
            sum_after = np.genfromtxt(matching_files_after[0], delimiter= ',',dtype= float)
    
    
            tot_trial_before[s]=sum_before
            tot_trial_after[s]=sum_after
    
    
    
           
        main_folder = r'F:/Videogame_Assay/'
        figure_folder_test = 'LFP_summary/'
        
        results_dir =os.path.join(main_folder + figure_folder_test)
        
        
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        
        
        
        csv_name_before =  RAT_ID[r] + '_'+band[b] + '_sum_before_'+event_name+'.csv'
        csv_name_after =  RAT_ID[r] + '_'+band[b] + '_sum_after_'+event_name+'.csv'   
        
        if len(sessions_subset)  == 5:
            
            np.savetxt(results_dir + csv_name_before, np.hstack((tot_trial_before[0],
                                                                 tot_trial_before[1],
                                                                 tot_trial_before[2],
                                                                 tot_trial_before[3],
                                                                 tot_trial_before[4],
                                                                   
                                                                   )), delimiter=',', fmt='%s') #final_trial_type (only for moving light)
         
            np.savetxt(results_dir + csv_name_after, np.hstack((tot_trial_after[0],
                                                                 tot_trial_after[1],
                                                                 tot_trial_after[2],
                                                                 tot_trial_after[3],
                                                                 tot_trial_after[4],
                                                                   
                                                                   )), delimiter=',', fmt='%s')          
            
            
            
                                                       
        else:
            
            np.savetxt(results_dir + csv_name_before, np.hstack((tot_trial_before[0],
                                                                 tot_trial_before[1],
                                                                 tot_trial_before[2],
                                                                 tot_trial_before[3],
                                                                                                                               
                                                                   )), delimiter=',', fmt='%s')
  

            np.savetxt(results_dir + csv_name_after, np.hstack((tot_trial_after[0],
                                                                 tot_trial_after[1],
                                                                 tot_trial_after[2],
                                                                 tot_trial_after[3],
                                                                                                                               
                                                                   )), delimiter=',', fmt='%s')         
            
            
##########################################
#stats on files with all the sessions
            
summary_folder = 'F:/Videogame_Assay/LFP_summary/'            
event_folder =  'touch/'       

    
band = ['delta','theta','beta','alpha']
       

for b in range(len(band)):
            
    for r in range(len(rat_summary_table_path)):       
        
        #session_path =  os.path.join(hardrive_path,session)    
        csv_dir_path = os.path.join(summary_folder + event_folder)
        offset_folder = 'pre/'
        csv_to_path = os.path.join(csv_dir_path + offset_folder)
           
        matching_files_before  = np.array(glob.glob(csv_to_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" ))
        sum_before = np.genfromtxt(matching_files_before[0], delimiter= ',',dtype= float)       
        
        offset_folder = 'post/'
        csv_to_path = os.path.join(csv_dir_path + offset_folder)
        
        matching_files_after = np.array(glob.glob(csv_to_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" ) )
        sum_after = np.genfromtxt(matching_files_after[0], delimiter= ',',dtype= float)
    
    
    #plot and save distribution before and after event
        plot_directory = 'F:/Videogame_Assay/LFP_summary_plots/'
        
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
            
            plt.hist(sum_before[channel,:],color= 'green', bins=50, alpha=.5,  linewidth=1,label='PRE touch')            
            plt.hist(sum_after[channel,:], color='red' ,bins=50,alpha=.5,  linewidth=1, label='POST touch')
                
                   
                
        plt.suptitle( RAT_ID[r]+ band[b] + event_folder[:-1])
        plt.legend(loc='upper right')  
        f0_title = 'distibution' +'_'+ band[b] +'_'+ RAT_ID[r] +'_'+ event_folder[:-1] + '.png'
        f0.tight_layout()
        f0.savefig(plot_directory+f0_title)
        plt.close()           
        
        
        #plot and save pvalues and bonferroni True/ False array 
        
        f1 =plt.figure(figsize=(20,10))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine() 
        
        
        #plot scatter plot using probe conf 

#        plt.scatter(x_coordinate_final,y_by_flatten_probe, c = probe_t_test, cmap="bwr",s=80,  edgecolors="k", linewidth=.2)
#        plt.colorbar()
#        plt.hlines(4808,0,12)

      
        
        indexes = [i for i,x in enumerate(bonferroni_annotation) if x == 1]
        no_indexes =[ele for ele in range(max(indexes)+1) if ele not in indexes]
        #len(indexes)+len(no_indexes)
        
        plt.scatter(x_coordinate_final[indexes],np.array(y_by_flatten_probe)[indexes], c =np.array( probe_t_test)[indexes], cmap="bwr", marker=(5, 2),s=60, linewidth=.5)
        
        plt.scatter(x_coordinate_final[no_indexes],np.array(y_by_flatten_probe)[no_indexes], c =np.array(probe_t_test)[no_indexes], cmap="bwr",s=60, edgecolors="k", linewidth=.2)
        
      
        plt.colorbar()
        plt.hlines(4808,0,12)
        plt.scatter(x_coordinate_final[c],np.array(y_by_flatten_probe)[c], c = 'k',s=60,  edgecolors="k", linewidth=.2)
       
    
#        
        #plot heatmap
        
        t_test_heatmap =np.reshape(probe_t_test,newshape=probe_map.shape) 
             
        ax = sns.heatmap(t_test_heatmap,annot=True,  cmap="bwr",vmin = 0, vmax=1,  edgecolors='white', linewidths=1,
                                      annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm() # "YlGnBu" RdBu
        
        ax.patch.set(hatch='//', edgecolor='black')
        
        
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.suptitle(RAT_ID[r]+ band[b] + event_folder[:-1])
        f1_title = 'p_value_heatmap' + band[b] +'_'+ RAT_ID[r] +'_'+ event_folder[:-1] + '.png'
        
        f1.savefig(plot_directory+f1_title)
        plt.close() 
        
        
        
        #save bonferroni and p value
        title = 'stats_'+ band[b] +'_'+ RAT_ID[r] +'_'+ event_folder[:-1] + '.csv'
        p_adjusted = multipletests(probe_t_test,alpha=.5, method='bonferroni')
        # 0.05/121 = 0.0004132231404958678
        np.savetxt(plot_directory + title, np.vstack((probe_t_test,p_adjusted[0])).T, delimiter=',', fmt='%s')
        print(r)
    print(band[b])






############################
    
#different stats and normalisation 
    

            
summary_folder = 'F:/Videogame_Assay/LFP_summary/'            
event_folder =  'reward/'       

    
band = ['delta','theta','beta','alpha']
       

for b in range(len(band)):
            
    for r in range(len(rat_summary_table_path)):       
        
        #session_path =  os.path.join(hardrive_path,session)    
        csv_dir_path = os.path.join(summary_folder + event_folder)
        offset_folder = 'pre/'
        csv_to_path = os.path.join(csv_dir_path + offset_folder)
           
        matching_files_before  = np.array(glob.glob(csv_to_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" ))
        sum_before = np.genfromtxt(matching_files_before[0], delimiter= ',',dtype= float)       
        
        offset_folder = 'post/'
        csv_to_path = os.path.join(csv_dir_path + offset_folder)
        
        matching_files_after = np.array(glob.glob(csv_to_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" ) )
        sum_after = np.genfromtxt(matching_files_after[0], delimiter= ',',dtype= float)
    
    
        ratio = sum_after/sum_before    
        #plot and save distribution before and after event

        probe_t_test= []
        
        
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten #probe_map_flatten
                         
            
            ch_t_test =  scipy.stats.ttest_1samp(ratio[ch], 1.0)
            probe_t_test.append(ch_t_test[1])
        
           
        
        #plot and save pvalues and bonferroni True/ False array 
        plot_directory = 'F:/Videogame_Assay/LFP_summary_plots/'
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
        plt.suptitle(RAT_ID[r]+ band[b] + event_folder[:-1])
        f1_title = 'p_value_heatmap_ttest_1samp' + band[b] +'_'+ RAT_ID[r] +'_'+ event_folder[:-1] + '.png'
        
        f1.savefig(plot_directory+f1_title)
        plt.close() 
        
                #save bonferroni and p value
        title = 'stats_ttest_1samp_'+ band[b] +'_'+ RAT_ID[r] +'_'+ event_folder[:-1] + '.csv'
        p_adjusted = multipletests(probe_t_test,alpha=.5, method='bonferroni')
        # 0.05/121 = 0.0004132231404958678
        np.savetxt(plot_directory + title, np.vstack((probe_t_test,p_adjusted[0])).T, delimiter=',', fmt='%s')
        print(r)
    print(band[b])

    
