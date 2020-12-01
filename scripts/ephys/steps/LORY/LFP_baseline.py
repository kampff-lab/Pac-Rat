# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:39:11 2020

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



RAT_ID = RAT_ID_ephys #[0]
rat_summary_table_path=rat_summary_ephys#[0]

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



        data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
        down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
        num_samples = int(int(len(down))/num_raw_channels)
        reshaped_down=  np.reshape(down,(num_samples,num_raw_channels))  
        down=None
        down_T = reshaped_down.T

        freq = 30000
        offset = 1500
        num_raw_channels = 128
        
        baseline_idx = np.arange(120000,num_samples-120000,6000) 


        
        csv_alpha_b = RAT_ID + '_sum_of_avg_alpha_base.csv'
        
        csv_beta_b = RAT_ID + '_sum_of_avg_beta_base.csv'

        csv_delta_b = RAT_ID + '_sum_of_avg_delta_base.csv'
   
        csv_theta_b = RAT_ID+ '_sum_of_avg_theta_base.csv' 
        
        
        
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten
            try:
                                
                
                ch_downsampled = down_T[channel,:]#should use channel
                #down_T=None
        
                baseline_chunk_around_event = np.zeros((len(baseline_idx),offset*2))
        
                   
                #baseline_chunk_around_event = ch_downsampled[baseline_idx-offset : baseline_idx+offset]
                for b, base in enumerate(baseline_idx):
                     
                    baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
                    #print(b)    
                print('epoch_baseline_DONE')
        

        
                p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
        
                p_base_avg = np.mean(p_base, axis =0) #[:len(downsampled_event_idx)]
                p_base_sem = stats.sem(p_base, axis = 0)
                
                # tot baseline excluding noise and sum 
                
                baseline_tot = [i for i,v in enumerate(f_base) if  v <45 or v>55 ]
                baseline_sel = p_base_avg[baseline_tot]
                baseline_sum_tot = np.sum(baseline_sel)
                tot_base_sum.append(baseline_sum_tot)         
                
                
                #baseline bands
                delta_ch_base = [i for i,v in enumerate(f_base) if 1< v <4 ]
                delta_sel_base = p_base_avg[delta_ch_base]
                delta_avg_base = np.mean(delta_sel_base)
                delta_sum_base = np.sum(delta_sel_base)
                delta_b.append(delta_avg_base)
                d_sum_base.append(delta_sum_base)
                
                theta_ch_base = [i for i,v in enumerate(f_base) if 4< v <8 ]
                theta_sel_base = p_base_avg[theta_ch_base]
                theta_avg_base = np.mean(theta_sel_base)
                theta_sum_base = np.sum(theta_sel_base)
                theta_b.append(theta_avg_base)
                t_sum_base.append(theta_sum_base)
                
                alpha_ch_base = [i for i,v in enumerate(f_base) if 8< v <12 ]
                alpha_sel_base = p_base_avg[alpha_ch_base]
                alpha_avg_base = np.mean(alpha_sel_base)
                alpha_sum_base = np.sum(alpha_sel_base)
                alpha_b.append(alpha_avg_base)
                a_sum_base.append(alpha_sum_base)
                
                beta_ch_base = [i for i,v in enumerate(f_base) if 12< v <30 ]
                beta_sel_base = p_base_avg[beta_ch_base]
                beta_avg_base = np.mean(beta_sel_base)
                beta_sum_base = np.sum(beta_sel_base)
                beta_b.append(beta_avg_base)
                b_sum_base.append(beta_sum_base)
  
            #np.savetxt(results_dir + csv_alpha, np.vstack(delta),delimiter=',', fmt='%s')
            
            except Exception:
                continue 
        #avg      
        alpha_rat[:,s]= alpha
        beta_rat[:,s] = beta
        delta_rat[:,s] = delta 
        theta_rat[:,s] =theta
        
        alpha_base[:,s]=alpha_b
        theta_base[:,s]=theta_b
        beta_base[:,s]=beta_b
        delta_base[:,s]=delta_b
        
        #sum
        alpha_rat_sum[:,s]= a_sum
        beta_rat_sum[:,s] = b_sum
        delta_rat_sum[:,s] = d_sum
        theta_rat_sum[:,s] = t_sum
        
        alpha_base_sum[:,s]= a_sum_base
        theta_base_sum[:,s]= t_sum_base
        beta_base_sum[:,s]= b_sum_base
        delta_base_sum[:,s]=d_sum_base    
                
        tot_base_sum_rat[:,s] = tot_base_sum
        tot_touch_sum_rat[:,s] = tot_touch_sum
        
        print('session_done')                
                
                