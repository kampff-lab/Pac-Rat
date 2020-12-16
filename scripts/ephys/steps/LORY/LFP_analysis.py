# -*- coding: utf-8 -*-
"""
Ephys Analysis: Step 1: downsample to 1 kHz

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

main_folder = 'E:/thesis_figures/'
figure_folder = 'Tracking_figures/'

results_dir =os.path.join(main_folder + figure_folder)


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#test ephys quality and pre processing on test clips from prior Trial end to current Trial end 

rat_summary_ephys = [r'F:/Videogame_Assay/AK_33.2_Pt.csv', 'F:/Videogame_Assay/AK_40.2_Pt.csv',
                          'F:/Videogame_Assay/AK_41.1_Pt.csv','F:/Videogame_Assay/AK_41.2_Pt.csv',
                              'F:/Videogame_Assay/AK_48.1_IrO2.csv','F:/Videogame_Assay/AK_48.4_IrO2.csv']


RAT_ID_ephys = ['AK 33.2', 'AK 40.2', 'AK 41.1', 'AK 41.2','AK 48.1','AK 48.4']


RAT_ID = RAT_ID_ephys [0]

hardrive_path = r'F:/'


#rat_summary_table_path = [r'F:/Videogame_Assay/AK_33.2_Pt.csv']

#s = len(rat_summary_table_path)


rat_summary_table_path=rat_summary_ephys[0]

#
#rat_summary_table_path = 'F:/Videogame_Assay/AK_33.2_Pt.csv'
#hardrive_path = r'F:/' 
#Level_2_post = prs.Level_2_post_paths(rat_summary_table_path)
#sessions_subset = Level_2_post

#
#
## Specify paths
#session  = sessions_subset[1]





probe_map_flatten = ephys.probe_map.flatten()
#new_probe_flatten=[103,7,21,90,75,30,1,123,88,17]




for r, rat in enumerate(rat_summary_table_path): 
    
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
        
        #recording data path
        #raw_recording = os.path.join(session_path +'/Amplifier.bin')
        downsampled_recording = os.path.join(session_path +'/Amplifier_downsampled.bin')
        #cleaned_recording =  os.path.join(session_path +'/Amplifier_cleaned.bin')
        
        #idx ro identify the start and the end of the clip of interest both in ephys samples and frames   
        csv_dir_path = os.path.join(session_path + '/events/')
        touch_path = os.path.join(hardrive_path, session +'/events/'+'RatTouchBall.csv')
        ball_on = os.path.join(hardrive_path, session +'/events/'+'BallON.csv')
        #trial_idx_path = os.path.join(csv_dir_path + 'Trial_idx.csv')
        trial_end_idx = os.path.join(csv_dir_path + 'TrialEnd.csv')
        #trial_idx = np.genfromtxt(trial_idx_path, delimiter = ',', dtype = int)
        
        video_csv = os.path.join(session_path + '/Video.csv')
        
        samples_for_frames_file_path = os.path.join(session_path + '/Analysis/samples_for_frames.csv')
        samples_for_frames = np.genfromtxt(samples_for_frames_file_path, dtype = int)
        



        trial_end = event_finder(trial_end_idx, video_csv, samples_for_frames_file_path)
        #trial prior end to current trial end based on ephys samples tp use with raw and cleaned recordings
        touching_light = event_finder(touch_path, video_csv, samples_for_frames_file_path)
        ball = event_finder(ball_on, video_csv, samples_for_frames_file_path)
        #generate random idx for baseline freq spectrum 
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
        
        start = 60000
        stop = len(reshaped_down)-start#offset*2
        idx = 2000 #len(touching_light)
        
        

        baseline_random = randint(start,stop,idx)
        baseline_idx = np.sort(baseline_random)
        #
        #test_baseline = downsampled_touch - baseline_idx
        #min_distance = np.min(abs(test_baseline))
        #max_distance = np.max(abs(test_baseline))
        #print(min_distance)
        #print(max_distance)
#            plt.figure()
#            plt.hist(baseline_random,bins=20)
#            
#            
#            #baseline_idx = downsampled_touch + 6000

        baseline_idx = np.arange(120000,num_samples-60000,6000)  
    
        #remove the first early trials
        downsampled_event_idx = downsampled_touch[1:]
        
        f0 =plt.figure(figsize=(20,20))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine() 
         
        
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten #probe_map_flatten
            #try:
                        
                
                ch_downsampled = down_T[channel,:]#should use channel
                #down_T=None
        
                chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
                
                baseline_chunk_around_event = np.zeros((len(baseline_idx),offset*2))
        
                for e, event in enumerate(downsampled_event_idx):
                     
                    chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
                    print(e)
                print('epoch_event_DONE')
        
           
                #baseline_chunk_around_event = ch_downsampled[baseline_idx-offset : baseline_idx+offset]
                for b, base in enumerate(baseline_idx):
                     
                    baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
                    #print(b)    
                print('epoch_baseline_DONE')
                    
                    
                ch_downsampled = None
                
                chunk_lenght = offset*2
                    
                p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
        
                p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
        
                
                test = np.sum(p_base)
                
                
                for t in arange(len(downsampled_event_idx)):
                    
                    if t < 5:
                        figure_name= str(t)+'.png'
                        plt.plot(f_base, p_base[t], color = 'k',alpha=1, label = 'touch', linewidth= 1)
                        plt.title('base')
                    else:
                        plt.plot(f_base, p_base[t], color = '#228B22',alpha=0.3, label = 'touch', linewidth= 1)
                        plt.title('base')
                    #plt.ylim(0,10e7)
                    #f0.savefig(results_dir + figure_name, transparent=True)
        
                plt.figure()
                for t in arange(len(downsampled_event_idx)):
                    
                    
                    figure_name= str(t)+'.png'
        
                    plt.plot(f_ch, p_ch[t], color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)
                    plt.title('touch')
                   # f0.savefig(results_dir + figure_name, transparent=True)
        
                
        
        
        
                p_ch_avg = np.mean(p_ch, axis =0)
                p_ch_sem = stats.sem(p_ch, axis = 0)
        
                p_base_avg = np.mean(p_base, axis =0) #[:len(downsampled_event_idx)]
                p_base_sem = stats.sem(p_base, axis = 0)

        
        
                ax = f0.add_subplot(5, 2, 1+ch, frameon=False)#all the probe is 11 11
                
                plt.figure()
                plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)
                plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                                 alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9) blue
        
                plt.figure()
                plt.plot(f_base, p_base_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
                plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
                                 alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')#green
               
                
                plt.xlim(0,100,2)
                plt.ylim(0,4e7)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                plt.xticks(fontsize=4, rotation=0)
                plt.yticks(fontsize=4, rotation=0)
                #plt.title('ch_'+ str(channel))
                #plt.legend(loc='best') 
                 
                    
            except Exception:
                continue 
               
        
         
        f0.subplots_adjust(wspace=.02, hspace=.02)
        figure_name =  RAT_ID + str(s) +'_freq_spectrum.pdf'
        f0.savefig(results_dir + figure_name, transparent=True)
        print('plot_saved')




######## avg 3 range of frequencies 



#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz
#gamma = 30-100 Hz
#high gamma = 60-100 Hz
        
probe_map_flatten = ephys.probe_map.flatten()
#new_probe_flatten=[103,7,21]


RAT_ID = RAT_ID_ephys [1]

rat_summary_table_path=rat_summary_ephys[1]


for r, rat in enumerate(rat_summary_table_path): 
    
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)
    
    alpha_rat =  np.zeros((N,tot_sessions))
    beta_rat = np.zeros((N,tot_sessions))
    theta_rat = np.zeros((N,tot_sessions))
    delta_rat = np.zeros((N,tot_sessions))
    
    alpha_base = np.zeros((N,tot_sessions))
    delta_base= np.zeros((N,tot_sessions))
    theta_base= np.zeros((N,tot_sessions))
    beta_base =  np.zeros((N,tot_sessions))
    
    alpha_rat_sum =  np.zeros((N,tot_sessions))
    beta_rat_sum = np.zeros((N,tot_sessions))
    theta_rat_sum = np.zeros((N,tot_sessions))
    delta_rat_sum = np.zeros((N,tot_sessions))
    
    alpha_base_sum = np.zeros((N,tot_sessions))
    delta_base_sum= np.zeros((N,tot_sessions))
    theta_base_sum= np.zeros((N,tot_sessions))
    beta_base_sum =  np.zeros((N,tot_sessions))    
    
    tot_base_sum_rat =  np.zeros((N,tot_sessions)) 
    tot_touch_sum_rat =  np.zeros((N,tot_sessions))
    
    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
        
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
        offset = 3000
        num_raw_channels = 128
        
        start = 60000
        stop = len(reshaped_down)-start#offset*2
        #idx = 2000 #len(touching_light)
        

        baseline_idx = np.arange(120000,num_samples-120000,6000)        


        #remove the first early trials
        downsampled_event_idx = downsampled_touch[1:]
        

         
        #delta = 1-4 Hz
        #theta = 4-8 Hz
        #alpha = 8-12 Hz
        #beta = 12-30 Hz        
        
        #saving means
        delta = []
        theta = []
        alpha = []
        beta = []
        
        alpha_b = []
        delta_b = []
        beta_b = []
        theta_b=[]
        
        csv_alpha = RAT_ID  +'_avg_alpha_touch.csv' #[r]
        
        csv_beta = RAT_ID +'_avg_beta_touch.csv'

        csv_delta = RAT_ID + '_avg_delta_touch.csv'
   
        csv_theta = RAT_ID +'_avg_theta_touch.csv'
        
        
        csv_alpha_b = RAT_ID + '_avg_alpha_base.csv'
        
        csv_beta_b = RAT_ID + '_avg_beta_base.csv'

        csv_delta_b = RAT_ID + '_avg_delta_base.csv'
   
        csv_theta_b = RAT_ID+ '_avg_theta_base.csv'       
        
        #saving sums
        
        a_sum =[]
        b_sum= []
        d_sum = []
        t_sum =[]
        
        a_sum_base =[]
        b_sum_base= []
        d_sum_base = []
        t_sum_base =[]
        
        
        tot_base_sum = []
        tot_touch_sum = []
        
        csv_alpha_sum = RAT_ID  +'_sum_alpha_touch.csv'
        
        csv_beta_sum = RAT_ID +'_sum_beta_touch.csv'

        csv_delta_sum = RAT_ID + '_sum_delta_touch.csv'
   
        csv_theta_sum = RAT_ID +'_sum_theta_touch.csv'
        
        
        csv_alpha_b_sum = RAT_ID + '_sum_alpha_base.csv'
        
        csv_beta_b_sum = RAT_ID + '_sum_beta_base.csv'

        csv_delta_b_sum = RAT_ID + '_sum_delta_base.csv'
   
        csv_theta_b_sum = RAT_ID+ '_sum_theta_base.csv'  
        
        
        csv_tot_baseline_sum = RAT_ID+ '_sum_tot_base.csv'  
          
        csv_tot_touch_sum =  RAT_ID+ '_sum_tot_touch.csv'
                
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten
            try:
                        
                
                ch_downsampled = down_T[channel,:]#should use channel
                #down_T=None
        
                chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
                
                baseline_chunk_around_event = np.zeros((len(baseline_idx),offset*2))
        
                for e, event in enumerate(downsampled_event_idx):
                     
                    chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
                    print(e)
                print('epoch_event_DONE')
        
           
                #baseline_chunk_around_event = ch_downsampled[baseline_idx-offset : baseline_idx+offset]
                for b, base in enumerate(baseline_idx):
                     
                    baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
                    #print(b)    
                print('epoch_baseline_DONE')
                    
                    
                ch_downsampled = None
                
                chunk_lenght = offset*2
                    
                p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
        
                p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
        
                
        
                p_ch_avg = np.mean(p_ch, axis =0)
                p_ch_sem = stats.sem(p_ch, axis = 0)
        
                p_base_avg = np.mean(p_base, axis =0) #[:len(downsampled_event_idx)]
                p_base_sem = stats.sem(p_base, axis = 0)


                touch_tot = [i for i,v in enumerate(f_ch) if  v <45 or v>55 ]
                touch_sel = p_ch_avg[touch_tot]
                touch_sum_tot = np.sum(touch_sel)
                tot_touch_sum.append(touch_sum_tot)
                
                
                # tot baseline excluding noise and sum 
                
                baseline_tot = [i for i,v in enumerate(f_base) if  v <45 or v>55 ]
                baseline_sel = p_base_avg[baseline_tot]
                baseline_sum_tot = np.sum(baseline_sel)
                tot_base_sum.append(baseline_sum_tot)
                
                #events bands
                delta_ch = [i for i,v in enumerate(f_ch) if 1< v <4 ]
                delta_sel = p_ch_avg[delta_ch]
                delta_avg = np.mean(delta_sel)
                delta_sum = np.sum(delta_sel)
                delta.append(delta_avg)
                d_sum.append(delta_sum)
                
                theta_ch = [i for i,v in enumerate(f_ch) if 4< v <8 ]
                theta_sel = p_ch_avg[theta_ch]
                theta_avg = np.mean(theta_sel)
                theta_sum = np.sum(theta_sel)
                theta.append(theta_avg)
                t_sum.append(theta_sum)
                
                alpha_ch = [i for i,v in enumerate(f_ch) if 8< v <12 ]
                alpha_sel = p_ch_avg[alpha_ch]
                alpha_avg = np.mean(alpha_sel)
                alpha_sum = np.sum(alpha_sel)
                alpha.append(alpha_avg)
                a_sum.append(alpha_sum)
                
                beta_ch = [i for i,v in enumerate(f_ch) if 12< v <30 ]
                beta_sel = p_ch_avg[beta_ch]
                beta_avg = np.mean(beta_sel)
                beta_sum= np.sum(beta_sel)
                beta.append(beta_avg)
                b_sum.append(beta_sum)
                
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
    
    

np.savetxt(results_dir + '_'+ csv_alpha, alpha_rat,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_beta, beta_rat,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_' + csv_delta, delta_rat,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_theta, theta_rat,delimiter=',', fmt='%s')



np.savetxt(results_dir + '_'+ csv_alpha_b, alpha_base,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_beta_b, beta_base,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_' + csv_delta_b, delta_base,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_theta_b, theta_base,delimiter=',', fmt='%s')


np.savetxt(results_dir + '_'+ csv_tot_baseline_sum, tot_base_sum_rat,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_tot_touch_sum, tot_touch_sum_rat,delimiter=',', fmt='%s')


np.savetxt(results_dir + '_'+ csv_alpha_sum, alpha_rat_sum,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_beta_sum, beta_rat_sum,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_' + csv_delta_sum, delta_rat_sum,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_theta_sum, theta_rat_sum,delimiter=',', fmt='%s')



np.savetxt(results_dir + '_'+ csv_alpha_b_sum, alpha_base_sum,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_beta_b_sum, beta_base_sum,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_' + csv_delta_b_sum, delta_base_sum,delimiter=',', fmt='%s')
np.savetxt(results_dir + '_'+ csv_theta_b_sum, theta_base_sum,delimiter=',', fmt='%s')











   
#
#
#        
#test = [i for i,v in enumerate(f_ch) if 1< v <4 ]
#test_avg = np.mean(p_ch_avg[test])
#
#plt.title('avg freq bands delta theta alpha beta')
#xcoords = [1, 4, 8,12,30,100]
#for xc in xcoords:
#    plt.axvline(x=xc,color='k')
#

########freq bands heatmap from.csv


# plot sum each freq baseline/tot baseline sum 

lfp_band = 'beta'


tot_base_file = 'E:/thesis_figures/Tracking_figures/tot_base/_'+RAT_ID+'_sum_tot_base.csv'
tot_sum = np.genfromtxt(tot_base_file, delimiter = ',', dtype = float) 

freq_band_file = 'E:/thesis_figures/Tracking_figures/' + lfp_band +'_base/_'+RAT_ID+'_sum_'+lfp_band+'_base.csv'
band = np.genfromtxt(freq_band_file, delimiter = ',', dtype = float) 

#session_tot = tot_sum[:,0]
#session_freq = band[:,0]
#
#rel_power = session_freq/session_tot
#
#probe_map =ephys.get_probe_map()
#N=121
##map the impedance
##probe_remap=np.reshape(probe_map, newshape=N)
##band_map =test[np.array(probe_remap)]
#band_final =np.reshape(rel_power,newshape=probe_map.shape)
#



f =plt.figure(figsize=(20,10))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)
#plt.title(title)
title = RAT_ID+'_'+lfp_band+ '_sum  base / sum tot base'
figure_name = RAT_ID+'_'+ lfp_band+ '_baseline_over_tot_baseline.png'

for i in arange(band.shape[1]):
    
    
   session_tot = tot_sum[:,i]
   session_freq = band[:,i]
   to_plot = session_freq/session_tot
   band_final =np.reshape(to_plot,newshape=probe_map.shape)


   ax = f.add_subplot(2,3, 1+i, frameon=True)
   plot = sns.heatmap(band_final,annot=False,  cmap="bwr", 
annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm()

#vmin = 0.01, vmax=.5


f.savefig(results_dir + figure_name, transparent=True)


###########################################

#lfp_band = 'delta'


touch_sum_file =  'E:/thesis_figures/Tracking_figures/' + lfp_band +'_touch/_'+RAT_ID+'_sum_'+lfp_band+'_touch.csv'
band_touch =  np.genfromtxt(touch_sum_file, delimiter = ',', dtype = float) 


tot_base_file = 'E:/thesis_figures/Tracking_figures/tot_base/_'+RAT_ID+'_sum_tot_base.csv'
tot_sum = np.genfromtxt(tot_base_file, delimiter = ',', dtype = float) 






f =plt.figure(figsize=(20,10))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)
#plt.title(title)
title = RAT_ID+'_'+lfp_band+ '_sum  touch/ sum tot base'
figure_name = RAT_ID+ '_'+ lfp_band+ '_touch_over_tot_baseline.png'

for i in arange(band.shape[1]):
    
    
   session_tot = tot_sum[:,i]
   session_freq_touch = band_touch[:,i]
   to_plot = session_freq_touch/session_tot
   band_final =np.reshape(to_plot,newshape=probe_map.shape)


   ax = f.add_subplot(2,3, 1+i, frameon=True)
   plot = sns.heatmap(band_final,annot=False,  cmap="bwr", 0, vmax=.5,
annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm()

#vmin = 0.01, vmax=.5


f.savefig(results_dir + figure_name, transparent=True)


####plot freq touch/freq base

#lfp_band = 'delta'


touch_sum_file =  'E:/thesis_figures/Tracking_figures/' + lfp_band +'_touch/_'+RAT_ID+'_sum_'+lfp_band+'_touch.csv'
band_touch =  np.genfromtxt(touch_sum_file, delimiter = ',', dtype = float) 



freq_band_file = 'E:/thesis_figures/Tracking_figures/' + lfp_band +'_base/_'+RAT_ID+'_sum_'+lfp_band+'_base.csv'
band = np.genfromtxt(freq_band_file, delimiter = ',', dtype = float) 



f =plt.figure(figsize=(20,10))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)
#plt.title(title)
title = RAT_ID+'_'+lfp_band+ '_sum  touch/ sum  base'
figure_name = RAT_ID+ '_'+lfp_band+ '_touch_over_baseline.png'

for i in arange(band.shape[1]):
    
    
   session_freq_base = band[:,i]
   session_freq_touch = band_touch[:,i]
   to_plot = session_freq_touch/session_freq_base
   band_final =np.reshape(to_plot,newshape=probe_map.shape)


   ax = f.add_subplot(2,3, 1+i, frameon=True)
   plot = sns.heatmap(band_final,annot=False,  cmap="bwr", vmin = 0, vmax=2,
annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm() # "YlGnBu" RdBu

#vmin = 0.01, vmax=.5

plt.title(title)
f.savefig(results_dir + figure_name, transparent=True)





#######################touch+/- 3s



        
probe_map_flatten = ephys.probe_map.flatten()
#new_probe_flatten=[103,7,21]


RAT_ID = RAT_ID_ephys [0]

rat_summary_table_path=rat_summary_ephys[0]


for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path
    Level_2_post = prs.Level_2_post_paths(rat)
    sessions_subset = Level_2_post
    
    N = 121
    tot_sessions = len(sessions_subset)
    
    alpha_rat_pre =  np.zeros((N,tot_sessions))
    beta_rat_pre = np.zeros((N,tot_sessions))
    theta_rat_pre = np.zeros((N,tot_sessions))
    delta_rat_pre = np.zeros((N,tot_sessions))
    
    alpha_rat_post =  np.zeros((N,tot_sessions))
    beta_rat_post = np.zeros((N,tot_sessions))
    theta_rat_post = np.zeros((N,tot_sessions))
    delta_rat_post = np.zeros((N,tot_sessions))
    

   
    
    for s, session in enumerate(sessions_subset):        
       
        
        session_path =  os.path.join(hardrive_path,session)
        
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
        
        event_name= 'reward.csv'
         
        #delta = 1-4 Hz
        #theta = 4-8 Hz
        #alpha = 8-12 Hz
        #beta = 12-30 Hz        
        
        #saving means
        delta_pre = []
        theta_pre = []
        alpha_pre = []
        beta_pre = []
        
        alpha_post = []
        delta_post = []
        beta_post = []
        theta_post=[]
        
        csv_alpha_pre = RAT_ID[r]  +'_alpha_before_' +event_name  #[r]
        
        csv_beta_pre = RAT_ID[r] +'_beta_before_' +event_name

        csv_delta_pre = RAT_ID[r] + '_delta_before_' +event_name
   
        csv_theta_pre = RAT_ID[r] +'_theta_before_' +event_name
        
        
        csv_alpha_post = RAT_ID[r] + '_alpha_after_' +event_name
        
        csv_beta_post = RAT_ID[r] + '_beta_after_' +event_name

        csv_delta_post = RAT_ID[r] + '_delta_after_' +event_name
   
        csv_theta_post = RAT_ID[r]+ '_theta_after_' +event_name     
        
        f0 =plt.figure(figsize=(20,20))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine() 
        figure_name =  RAT_ID[r] + '_'+ session[-16:] + '_pre_post.png'
                
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
        
                
        
                p_before_avg = np.mean(p_before, axis =0)
                p_before_sem = stats.sem(p_before, axis = 0)
        
                p_after_avg = np.mean(p_after, axis =0) #[:len(downsampled_event_idx)]
                p_after_sem = stats.sem(p_after, axis = 0)



                ax = f0.add_subplot(11, 11, 1+ch, frameon=False)#all the probe is 11 11
                

                #plt.figure()
                plt.plot(f_before, p_before_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)
                plt.fill_between(f_before, p_before_avg-p_before_sem, p_before_avg+p_before_sem,
                                 alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9) blue
        
                #plt.figure()
                plt.plot(f_after, p_after_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
                plt.fill_between(f_after, p_after_avg-p_after_sem, p_after_avg+p_after_sem,
                                 alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')#green
      

                
                #events bands BEFORE
                
                delta_ch = [i for i,v in enumerate(f_before) if 1< v <4 ]
                delta_sel = p_before_avg[delta_ch]        
                delta_sum = np.sum(delta_sel)
                delta_pre.append(delta_sum)
                
                theta_ch = [i for i,v in enumerate(f_before) if 4< v <8 ]
                theta_sel = p_before_avg[theta_ch]
                theta_sum = np.sum(theta_sel)
                theta_pre.append(theta_sum)
                
                alpha_ch = [i for i,v in enumerate(f_before) if 8< v <12 ]
                alpha_sel = p_before_avg[alpha_ch]
                alpha_sum = np.sum(alpha_sel)           
                alpha_pre.append(alpha_sum)
                
                beta_ch = [i for i,v in enumerate(f_before) if 12< v <30 ]
                beta_sel = p_before_avg[beta_ch]             
                beta_sum= np.sum(beta_sel)             
                beta_pre.append(beta_sum)

                #events bands AFTER
                
                delta_ch = [i for i,v in enumerate(f_after) if 1< v <4 ]
                delta_sel = p_after_avg[delta_ch]            
                delta_sum = np.sum(delta_sel)             
                delta_post.append(delta_sum)
                
                theta_ch = [i for i,v in enumerate(f_after) if 4< v <8 ]
                theta_sel = p_after_avg[theta_ch]              
                theta_sum = np.sum(theta_sel)             
                theta_post.append(theta_sum)
                
                alpha_ch = [i for i,v in enumerate(f_after) if 8< v <12 ]
                alpha_sel = p_after_avg[alpha_ch]
                alpha_sum = np.sum(alpha_sel)               
                alpha_post.append(alpha_sum)
                
                beta_ch = [i for i,v in enumerate(f_after) if 12< v <30 ]
                beta_sel = p_after_avg[beta_ch]               
                beta_sum= np.sum(beta_sel)               
                beta_post.append(beta_sum)


            except Exception:
                continue 
             
        alpha_rat_pre[:,s]= alpha_pre
        beta_rat_pre[:,s] = beta_pre
        delta_rat_pre[:,s] = delta_pre
        theta_rat_pre[:,s] =theta_pre
        
  
        alpha_rat_post[:,s]= alpha_post
        beta_rat_post[:,s] = beta_post
        delta_rat_post[:,s] = delta_post
        theta_rat_post[:,s] = theta_post
        
        plt.tight_layout()



        f0.savefig(results_dir + figure_name, transparent=False)
        plt.close()                
                                
        print('session_done')
    

    results_dir =os.path.join(main_folder + figure_folder + 'pre_reward/')
    
    
    #pre touch
    np.savetxt(results_dir + '_'+ csv_alpha_pre, alpha_rat_pre,delimiter=',', fmt='%s')
    np.savetxt(results_dir + '_'+ csv_beta_pre, beta_rat_pre,delimiter=',', fmt='%s')
    np.savetxt(results_dir + '_' + csv_delta_pre, delta_rat_pre,delimiter=',', fmt='%s')
    np.savetxt(results_dir + '_'+ csv_theta_pre, theta_rat_pre,delimiter=',', fmt='%s')
    
    
    results_dir =os.path.join(main_folder + figure_folder + 'post_reward/')
    #pos touch
    np.savetxt(results_dir + '_'+ csv_alpha_post, alpha_rat_post,delimiter=',', fmt='%s')
    np.savetxt(results_dir + '_'+ csv_beta_post, beta_rat_post,delimiter=',', fmt='%s')
    np.savetxt(results_dir + '_' + csv_delta_post, delta_rat_post,delimiter=',', fmt='%s')
    np.savetxt(results_dir + '_'+ csv_theta_post, theta_rat_post,delimiter=',', fmt='%s')
    
















#plot before/post touch


#RAT_ID = RAT_ID_ephys [0]

#rat_summary_table_path=rat_summary_ephys[0]




lfp_band = ['alpha','beta','delta','theta']

for rat in range(len(RAT_ID)):

    
    for b, band in enumerate(lfp_band):
        
        
        
        before_touch_file =  'E:/thesis_figures/Tracking_figures/pre_ball_on/_'+RAT_ID[rat] + '_'+ band+ '_before_ball.csv'
        before_touch =  np.genfromtxt(before_touch_file, delimiter = ',', dtype = float) 
        
        
        after_touch_file = 'E:/thesis_figures/Tracking_figures/post_ball_on/_'+RAT_ID[rat] + '_'+ band+ '_after_ball.csv'
        after_touch = np.genfromtxt(after_touch_file, delimiter = ',', dtype = float) 
        
        
            
        Level_2_post = prs.Level_2_post_paths(rat_summary_table_path[rat])
        sessions_subset = Level_2_post   
         
        
        f =plt.figure(figsize=(20,10))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=False)
        #plt.title(title)
        title = RAT_ID[rat]+'_'+band+ '_post/pre_ball_on'
        figure_name = RAT_ID[rat]+ '_'+band+ '_post_over_pre_ball_on.png'
        
            
        for s, session in enumerate(sessions_subset):   
        
            session_path =  os.path.join(hardrive_path,session)     
            csv_dir_path = os.path.join(session_path +'/bad_channels.csv')
            
            bad_ch = np.genfromtxt(csv_dir_path, delimiter = ',', dtype=int)
        
            session_freq_before = before_touch[:,s]
            session_freq_after = after_touch[:,s]
                
            to_plot = session_freq_after/session_freq_before
        
            c = np.array(bad_ch.astype(int).tolist())
            
            to_plot[c]=np.nan
            
            band_final =np.reshape(to_plot,newshape=probe_map.shape)
        
        
            ax = f.add_subplot(2,3, 1+s, frameon=True)
            ax = sns.heatmap(band_final,annot=False,  cmap="bwr", vmin = 0.75, vmax=1.25, edgecolors='white', linewidths=1,
                              annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm() # "YlGnBu" RdBu
            ax.patch.set(hatch='//', edgecolor='black')
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
        
        plt.title(title)
        f.savefig(results_dir + figure_name, transparent=False)
        print(band)
    print(session)
print(rat)
    
    







































        
    
    
for i, idx in enumerate(chunk_around_event):
    plt.figure()
    plt.plot(idx)
    plt.title('i')
    
for i, idx in enumerate(baseline_chunk_around_event):
    plt.figure()
    plt.plot(idx)
    plt.title('i')
     
    
#create epochs 
test_epochs= None
        
data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(down))/num_raw_channels)
reshaped_down=  np.reshape(down,(num_samples,128))  
down=None
down_T = reshaped_down.T

downsampled_event_idx = downsampled_touch[1:]
test_epochs = np.zeros((len(downsampled_event_idx), len(probe_map_flatten),offset*2))  
#new_probe_flatten_test = [103,7,21,90,75,30]    
for ch, channel in enumerate(probe_map_flatten):
    try:
        
        
       
        ch_downsampled = down_T[ch,:]
        #down_T=None

        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        #baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)

        test_epochs[:,ch,:] = chunk_around_event
        print(ch)
        #baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))


        #for b, base in enumerate(baseline_idx):
   
            #baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
            #print(b)
            

            
    except Exception:
        continue     
    
################################

test= None
    
freqs = np.arange(3.0, 100.0, 2.0)    
    
test_tfr = time_frequency.tfr_array_multitaper(test_epochs,sfreq= 1000,freqs = freqs, output= 'avg_power',n_jobs=8)   

norm = np.mean(test_tfr2[94,:20,:1000],axis=1)

norm_expanded = np.repeat([norm], offset*2, axis=0).T

ch_test_norm = test_tfr2[94,:20,:]/norm_expanded


ch_test = np.log(test_tfr[94,:20,:])
plt.figure()
plt.imshow(np.flipud(ch_test_norm),aspect='auto', cmap='jet')#,vmin=0.4, vmax =1.9)
#plt.axvline(6000,20,color='k')
plt.colorbar()

   
f0 =plt.figure(figsize=(20,20))
#outer_grid = gridspec.GridSpec(11, 11, wspace=0.0, hspace=0.0)

for i, ch in enumerate(probe_map_flatten):
    #inner_grid = gridspec.GridSpecFromSubplotSpec(1, 1,
     #       subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

    norm = np.mean(test_tfr[i,:20,:1000],axis=1)
    norm_expanded = np.repeat([norm], offset*2, axis=0).T
    ch_test_norm = test_tfr[i,:20,:]/norm_expanded
    ch_test = np.log(test_tfr[i,:20,:])
       
    ax = f0.add_subplot(11, 11, 1+i, frameon=False)

    plot = ax.imshow(np.flipud(ch_test_norm),aspect='auto', cmap='jet')#,vmin=0.4, vmax =1.9)
    cbar=plt.colorbar(plot,fraction=0.04, pad=0.04, aspect=10, orientation='horizontal')
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=5)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
   
f0.subplots_adjust(wspace=.02, hspace=.02)
plt.title('raw')

#
#test_plot 
#
#event = downsampled_end[20]
#
#
#chunk_around_event_raw = np.zeros((len(probe_map_flatten),offset*2))
#        
#       
#    for c, ch in enumerate(probe_map_flatten):
#             
#        chunk_around_event_raw[c,:] = ch_downsampled[event-offset : event+offset]
#            print(e)
#
#
#
#


#test plot
channel = 103


raw_data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(raw_data))/num_raw_channels)

# Reshape data to have 128 rows
reshaped_raw_data = np.reshape(raw_data,(num_samples,num_raw_channels)).T
raw = reshaped_raw_data[channel, :]
       
ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
raw = None
       
        
ch_lowpass = butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
        
        
plt.figure()
plt.plot(ch_lowpass[30000:45000])
plt.title('lowpass')
        
ch_downsampled = ch_lowpass[::30]        
 










#test different trial lenghts



freq = 30000
num_raw_channels = 128
final_trial_length = 8192 #(power of 2)
intermediate_trial_length = 9000

ball=ball[:-1]

data = np.memmap(cleaned_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
#down_sample_lenght = num_samples/30

probe_map_flatten = ephys.probe_map.flatten()
channel = 103
raw = reshaped_data[channel, :]
      
# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
raw=None
ch_lowpass = butter_filter_lowpass(ch_raw_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')


n = len(touching_light)   
ball_to_touch_chunks = [[] for _ in range(n)] 
touch_to_reward =  [[] for _ in range(n)] 



for trial in np.arange(n):
    b = ball[trial]
    t = touching_light[trial]
    e = trial_end[trial]
    ball_to_touch_chunks[trial]= ch_lowpass[b:t]
    touch_to_reward[trial]=ch_lowpass[t:e]
    

test= ball_to_touch_chunks.pop(2)
test2=ball_to_touch_chunks.pop(14)
test3=touch_to_reward.pop(2)
test4 = touch_to_reward.pop(14)


downsampled_ball_to_touch_chunks =  [[] for _ in range(len(ball_to_touch_chunks))] 
downsampling_factor_bt = []

for t, trial in enumerate(ball_to_touch_chunks):
    trial_length = len(trial)
    down = np.uint32((trial_length/intermediate_trial_length))
    downsampling_factor_bt.append(down)
    t_down = trial[::down]
    length_down = len(t_down)
    diff_from_final = length_down - final_trial_length
    downsampled_ball_to_touch_chunks[t]=t_down[:-diff_from_final]






downsampled_touch_to_reward_chunks =  [[] for _ in range(len(touch_to_reward))] 
downsampling_factor_tr = []


for t, trial in enumerate(touch_to_reward):
    trial_length = len(trial)
    down = np.uint32((trial_length/intermediate_trial_length))
    downsampling_factor_tr.append(down)
    t_down = trial[::down]
    length_down = len(t_down)
    diff_from_final = length_down - final_trial_length
    downsampled_touch_to_reward_chunks[t]=t_down[:-diff_from_final]




downsampled_1000_touch_to_reward_chunks =  [[] for _ in range(len(touch_to_reward))] 


for t, trial in enumerate(touch_to_reward):

    t_down = trial[::30]
    downsampled_1000_touch_to_reward_chunks[t]=t_down




#test_trial =downsampled_touch_to_reward_chunks[24]
#len(test_trial)
#
#
#
#
#long_trial = []
#
#
#for f, factor in enumerate(downsampling_factor):
#
#    long = [idx for idx, val in factor) if val > 9000000 ]#or val < 150000] 
#    print (min(mean_impedance_Level_2_post[s]))
#    print (max(mean_impedance_Level_2_post[s]))
#    if idx_bad_imp == 0 :
#        
#        bad_channels_idx[s] = []
#    else:
#       bad_channels_idx[s] = idx_bad_imp 
#
    
    
    
    
    
    
p_all = []
f_all = []

for t,test_trial in enumerate(downsampled_1000_touch_to_reward_chunks):

    band = 5000/len(test_trial)
    print(band)

    p_ch, f_ch = time_frequency.psd_array_multitaper(test_trial, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = band, n_jobs = 8)
    p_all.append(p_ch)
    f_all.append(f_ch)


for p in np.arange(len(downsampled_touch_to_reward_chunks)):
    plt.figure()
    plt.plot(f_all[1],p_all[1])
    plt.plot(f_all[13],p_all[13])
    plt.plot(f_all[40],p_all[40])
    plt.plot(f_all[7],p_all[7])


avg = np.mean(p_all)


#6000 2.5
#10551 1.5
# 26586 1.5


plt.figure()
plt.plot(f_ch,p_ch)



p_all = []
f_all = []

for t in downsampled_ball_to_touch_chunks:
    
    
    p_ch, f_ch = time_frequency.psd_array_multitaper(t, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
    p_all.append(p_ch)
    f_all.append(f_ch)




plt.figure()
plt.plot(f_ch,p_ch)



#11x11 freq spectra plot around events

freq = 30000
offset = 3000
num_raw_channels = 128


data = np.memmap(downsampled_recording, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_raw_channels)
reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
down_sample_lenght = num_samples/30


#20 sec into the session

baseline_idx = 23000

# event before baseline

baseline = 



probe_map_flatten = ephys.probe_map.flatten()
new_probe_flatten_test =[103,7,21,90,75,30]


#remove the first early trials
downsampled_event_idx = downsampled_end[1:]

f0 =plt.figure(figsize=(20,20))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
 

for ch, channel in enumerate(probe_map_flatten):
    try:
                
        #data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        raw = reshaped_data[channel, :]
        #reshaped_data = None
        
        # Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
        ch_raw_uV = (raw.astype(np.float32) - 32768) * 0.195
        raw = None
        ch_downsampled=ch_raw_uV
                
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        


        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        baseline_chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)


   
        baseline_chunk_around_event = ch_downsampled[baseline_idx-offset : baseline_idx+offset]
            
            
            
        ch_downsampled = None
        
        chunk_lenght = offset*2
            
        p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)

        p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)


        p_ch_avg = np.mean(p_ch, axis =0)
        p_ch_sem = stats.sem(p_ch, axis = 0)




        ax = f0.add_subplot(11, 11, 1+ch, frameon=False)
        
        #plt.figure()
        plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)
        plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9)

        
        plt.plot(f_base, p_base, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
        #plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
                         #alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')
       
        plt.ylim(0,300000)
        plt.xticks(fontsize=4, rotation=0)
        plt.yticks(fontsize=4, rotation=0)
        #plt.title('ch_'+ str(channel))
        #plt.legend(loc='best') 
         
            
    except Exception:
        continue 
       

 
f0.subplots_adjust(wspace=.02, hspace=.02)


###############################################################################################

downsampled_ball= downsampled_ball[:-1]

#previous event as baseline 
freq = 30000
offset = 3000
num_raw_channels = 128




data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(down))/num_raw_channels)
reshaped_down=  np.reshape(down,(num_samples,128))  
down=None
down_T = reshaped_down.T
#ch_downsampled = down_T[channel,:]
#down_T=None

down_sample_lenght = num_samples/30

baseline_event_idx = downsampled_end[1:]

#remove the first early trials
downsampled_event_idx = downsampled_ball[1:]

f0 =plt.figure(figsize=(20,20))
sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine() 
 
probe_map_flatten = ephys.probe_map.flatten()
for ch, channel in enumerate(probe_map_flatten):
    try:
                
        #data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
        #num_samples = int(int(len(data))/num_raw_channels)

        # Reshape data to have 128 rows
        #reshaped_data = np.reshape(data,(num_samples,num_raw_channels)).T
        #data = None
        
        # Extract selected channel (using probe map)
        # = probe_map[depth, shank]
        ch_downsampled = down_T[ch,:]
        #reshaped_data = None
        
       
        #plt.figure()
        #plt.plot(ch_downsampled[1000:1500])
        


        chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))
        
        baseline_chunk_around_event = np.zeros((len(downsampled_event_idx),offset*2))

        for e, event in enumerate(downsampled_event_idx):
             
            chunk_around_event[e,:] = ch_downsampled[event-offset : event+offset]
            print(e)


   
        baseline_chunk_around_event = np.zeros((len(baseline_event_idx),offset*2))


        for b, base in enumerate(baseline_event_idx):
   
            baseline_chunk_around_event[b,:] = ch_downsampled[base-offset : base+offset]
            print(b)
            

            
            
        ch_downsampled = None
        
        chunk_lenght = offset*2
            
        p_ch, f_ch = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 60, bandwidth = 2.5, n_jobs = 8)

        p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 60, bandwidth = 2.5, n_jobs = 8)


        p_ch_avg = np.mean(p_ch, axis =0)
        p_ch_sem = stats.sem(p_ch, axis = 0)

 
        p_base_avg = np.mean(p_base, axis =0)
        p_base_sem = stats.sem(p_base, axis=0)



        ax = f0.add_subplot(11, 11, 1+ch, frameon=False)
        
        #plt.figure()
        plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'event', linewidth= 1)
        plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9)

        #plt.figure()
        plt.plot(f_base, p_base_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)   
        plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
                         alpha=0.4, edgecolor='#228B22', facecolor='#32CD32')
       
        plt.ylim(0,300000)
        plt.xticks(fontsize=4, rotation=0)
        plt.yticks(fontsize=4, rotation=0)
        #plt.title('ch_'+ str(channel))
   
            
    except Exception:
        continue 



#f0.legend(loc='best')  
f0.subplots_adjust(wspace=.02, hspace=.02)

bad_trials = []

for p, power in enumerate(p_base):
    max_value = max(power)
    bad_trials.append(max_value)
        

bad_t = []       
for b,bad in enumerate(bad_trials):
    if bad >=10000000.0:
        bad_t.append(b)
        



for t,trial in enumerate(p_base_final):
    plt.figure()
    plt.plot(f_base,trial)
    #plt.plot(f_ch, p_ch[t])
    #plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
                         #alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9)
    

test = np.delete(test,15,0)
test = p_base_final




        p_base_avg_t = np.mean(test, axis =0)
        p_base_sem_t = stats.sem(test, axis=0)
        plt.figure()
        plt.plot(f_base, p_base_avg_t, color = '#1E90FF',alpha=0.3, label = 'event', linewidth= 1)
        plt.fill_between(f_base, p_base_avg_t-p_base_sem_t, p_base_avg_t+p_base_sem_t,
                         alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')








min_freq = 10
max_freq = 30

frequency_finder = [index for index,value in enumerate(f_ch) if value >= min_freq and value <= max_freq]

min_range = min(frequency_finder)
max_range = max(frequency_finder)

power_finder_in_range = p_ch[min_range:max_range]

avg_power = np.mean(power_finder_in_range)






























####test####################################################################################














p, f = time_frequency.psd_array_multitaper(ch_lowpass[15000:30000], sfreq= 30000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#
#plt.figure()
#plt.plot(f,p)
#
#
#
#pd, fd = time_frequency.psd_array_multitaper(data_downsampled[500:1000], sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#plt.figure()
#plt.plot(fd,pd)











#create a list if tuples (end, lenght) for each trials excluding the first one

#count = 0
#trial_end_and_lenght = []
#
#for idx in np.arange(len(end_samples)):
#    
#    if idx == 0:
#        count =+ 1
#    else:          
#        start_sample = end_samples[idx-1]
#        num_samples = samples_lenght_end_to_end[idx]
#        trial_end_and_lenght.append((start_sample, num_samples))
#
#



















        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
test =  stats.sem(p_ch, axis = 0)
test_b = stats.sem(p_base)

std = p_ch_std
std_base = p_base_std

plt.plot(x, y, 'k-')
plt.fill_between(x, y-error, y+error)
plt.show()

plt.plot(f_ch, p_ch_avg, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)    
plt.fill_between(f_ch, p_ch_avg-p_ch_sem, p_ch_avg+p_ch_sem,
    alpha=0.5, edgecolor='#1E90FF', facecolor='#00BFFF')

plt.plot(f_base, p_base_avg, color = '#228B22',alpha=0.3,  label = 'baseline', linewidth= 1)    
plt.fill_between(f_base, p_base_avg-p_base_sem, p_base_avg+p_base_sem,
    alpha=0.5, edgecolor='#228B22', facecolor='#32CD32')
    
plt.legend()   
 




   
plt.figure()
plt.plot(f_ch,p_ch_avg[:])
plt.title('ch_'+str(channel))


sns.set()



p_avg_base = np.mean(p_base, axis =0)
f_avg_base  = np.mean(f_base)


sns.set()

#plt.figure()
plt.plot(f_base,p_base_avg[:],'r')
plt.title('baseline_ch_'+str(channel))


           
            
            
            
            
    except Exception:
        continue 


sns.set()


# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')

    
    
    

#num_channels = 128
#data = np.memmap(raw_recording, dtype = np.uint16, mode = 'r')
#num_samples = int(int(len(data))/num_channels)
#freq = 30000
#recording_time_sec = num_samples/freq
#recording_time_min = recording_time_sec/60
#reshaped_data = np.reshape(data,(num_samples,128))
##to have 128 rows
#reshaped_data_T= reshaped_data.T
#data = None
#
#
##signal_reshaped = ephys.apply_probe_map_to_amplifier(reshaped_data_T)
##
## Extract data chunk for single channel
#channel = 4
#
#channel_data = reshaped_data_T[channel,:]
#reshaped_data_T = None


#ch_mean = np.mean(channel_data, axis=0)

#ch_std = np.std(channel_data, axis=0)

#channel_data_Z = channel_data - ch_mean



# Z-score each channel



#raw_Z = np.zeros(raw_uV.shape)
#for ch in range(128):
#    raw_Z[ch,:] = (raw_uV[ch,:] - raw_mean[ch]) / raw_std[ch]

# Specify channels to exclude

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
channel_data = None





data_lowpass = butter_filter_lowpass(channel_data_uV, lowcut=250,  fs=30000, order=3, btype='lowpass')
#channel_data_highpass = butter_filter(channel_data_uV, 500, 5000, fs=30000, order=3, btype='bandpass')
plt.figure()
plt.plot(data_lowpass[30000:45000])



data_downsampled = data_lowpass[::30]

plt.figure()
plt.plot(data_downsampled[1000:1500])


#
##working test mne fx for multitaper 
#        
#   
#
p, f = time_frequency.psd_array_multitaper(ch_lowpass[15000:30000], sfreq= 30000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#
#plt.figure()
#plt.plot(f,p)
#
#
#
#pd, fd = time_frequency.psd_array_multitaper(data_downsampled[500:1000], sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 10, n_jobs = 8)
#plt.figure()
#plt.plot(fd,pd)


offset = 3000

downsampled_touch = np.uint32(np.array(touching_light)/30)

chunk_around_event = np.zeros((len(downsampled_touch),offset*2))

for e, event in enumerate(downsampled_touch):
    try:  
        chunk_around_event[e,:] = data_downsampled[event-offset : event+offset]
        print(e)
    except Exception:
        continue



baseline_chunk_around_event = np.zeros((len(downsampled_touch),offset*2))


for b, base in enumerate(baseline_idx):
    try:  
        baseline_chunk_around_event[b,:] = data_downsampled[base-offset : base+offset]
        print(b)
    except Exception:
        continue 










chunk_lenght = offset*2

p_test, f_test = time_frequency.psd_array_multitaper(chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)

p_base, f_base = time_frequency.psd_array_multitaper(baseline_chunk_around_event, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)




plt.figure()
plt.plot(f_ch,p_ch[40,:],label='1')
plt.label()

#plt.figure()
plt.plot(f_base,p_base[40,:],'r')





p_avg = np.mean(p_test, axis =0)
f_avg  = np.mean(f_test)



plt.figure()
plt.plot(f_test,p_avg[:])
plt.title('ch_'+str(channel))




p_avg_base = np.mean(p_base, axis =0)
f_avg_base  = np.mean(f_base)



#plt.figure()
plt.plot(f_base,p_avg_base[:],'r')
plt.title('baseline_ch_'+str(channel))
























































#samples_fft = np.fft.rfft(chunk_around_event)
#frequencies = np.abs(samples_fft)
freq_mean = np.mean(frequencies, axis=0)
plt.plot(freq_mean[:100])




#f, t, Sxx = signal.spectrogram(chunk_around_event, 1000, nperseg=1000, nfft=1000, noverlap=500)

### lowpass filter LFP

lowcut = 250

lowpass_data = np.zeros((len(probe_Z),num_samples))
lowpass_downsampled = [[] for _ in range(len(probe_Z))]  

for channel in np.arange(len(probe_Z)):
    try:  
        channel_data = probe_Z[channel,:]
        lowpass_cleaned = ephys.butter_filter_lowpass(channel_data,lowcut, fs=30000, order=3, btype='lowpass')
        downsampled_ch = lowpass_cleaned[::30]
        lowpass_data[channel,:] = lowpass_cleaned
        lowpass_downsampled[channel] = downsampled_ch
        print(channel)
        
    except Exception:




        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 f, t, Zxx = signal.stft(lowpass_cleaned, fs, nperseg=20000)

plt.pcolormesh(t, f, np.abs(Zxx))


# Downsample each channel
num_ds_samples = np.int(np.floor(num_samples / 30))
downsampled = np.zeros((128, num_ds_samples))
for ch in range(128):
    raw_ch = raw[ch,:]
    lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
    downsampled_ch = lowpass_ch[::30]
    downsampled[ch, :] = downsampled_ch[:num_ds_samples]




lowpass_data[22,:]

# Store downsampled data in a binary file







# Report
ch = 21
raw_ch = raw[ch,:]
lowpass_ch = ephys.butter_filter_lowpass(raw_ch, 500)
downsampled_ch = downsampled[ch, :]
plt.figure()
plt.plot(raw_ch, 'r')
plt.plot(lowpass_ch, 'g')
plt.plot(np.arange(num_ds_samples) * 30, downsampled_ch, 'b')
plt.show()

# LORY (spectral analysis, LFP, etc.)

#FIN    
        
#2 and 65 opposite phase        
        
        
plt.plot(lowpass_data[100,:150000],alpha = 0.4)




##### downsampling from 30kHz to 1kHz


# Spectrogram test
plt.figure()
shank = 4
for depth in range(11):
    plt.subplot(11,2,depth*2 + 1)
    probe_Z = ephys.apply_probe_map_to_amplifier(clean_Z)
    fs = 30000
    ch = (depth * 11) + shank
    f, t, Sxx = signal.spectrogram(probe_Z[ch,:], fs, nperseg=30000, nfft=30000, noverlap=27000)
    plt.pcolormesh(t, f, Sxx)
    plt.ylim([0, 30])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(11,2,depth*2 + 2)
    plt.plot(probe_Z[ch,:])
plt.show()

#FIN