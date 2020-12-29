# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:25:23 2020

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

# create raw downsampld chunks around event for each channel all the trial and save a cube ch x samples x trials

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
        
      
 
        matrix_around_event = np.zeros((121,offset*2,len(downsampled_event_idx)))

            
            
        
        csv_name = RAT_ID[r]+ '_sum_of_avg_alpha_base.csv'
        
        csv_beta_b = RAT_ID[r] + '_sum_of_avg_beta_base.csv'

        csv_delta_b = RAT_ID[r] + '_sum_of_avg_delta_base.csv'
   
        csv_theta_b = RAT_ID[r]+ '_sum_of_avg_theta_base.csv' 
        


        
        for ch, channel in enumerate(probe_map_flatten): #new_probe_flatten probe_map_flatten
           
                        
                
            ch_downsampled = down_T[channel,:]#should use channel
            #down_T=None
    
            chunk = np.zeros((len(downsampled_event_idx),offset*2))
                         
    
            for e, event in enumerate(downsampled_event_idx):
                 
                chunk[e,:] = ch_downsampled[event-offset : event+offset]
                
                
                print(e)
            matrix_around_event[ch,:,:]  = chunk.T

         filename = 'all_ch_around_touch'
         saving_folder = os.path.join(session_path + '/LFP/' + filename )
         np.save(saving_folder,matrix_around_event)       



test = np.load('F:/Videogame_Assay/AK_33.2/2018_04_28-16_26/LFP/all_ch_around_touch.npy')       
        
     

dt = 0.001                                # Define the sampling interval.
K = np.shape(test)[2]               # Define the number of trials.
N = np.shape(test)[1]             # Define number of points in each trial.

f = rfftfreq(N, dt)   

#Visualizing the Phase Difference across Trials

j8 = where(f==30)[0][0]       # Determine index j for frequency 10 Hz.
j24 = where(f==10)[0][0]     # Determine index j for frequency 50 Hz.

phi8 = zeros(K)              # Variables to hold phase differences.
phi24 = zeros(K)

ch_1 =100
ch_2 = 15
T = 3 

#ch deve  essere trialsxlenght
for k in range(K):           # For each trial, compute the cross spectrum. 
    x =test[ch_1].T[k] - mean(test[ch_1].T[k])  # Get the data from each electrode,
    y = test[ch_2].T[k].T - mean(test[ch_2].T[k,:])
    xf = rfft(x - mean(x))   # ... compute the Fourier transform,
    yf = rfft(y - mean(y))
    Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    phi8[k] = angle(Sxy[j8]) # ... and the phases.
    phi24[k] = angle(Sxy[j24])
                             # Plot the distributions of phases.
_, (a1, a2) = subplots(1, 2, sharey=True, sharex=True)
a1.hist(phi8, bins=20, range=[-pi, pi])
a2.hist(phi24, bins=20, range=[-pi, pi])

#ylim([0, 40])                # Set y-axis and label axes.
a1.set_ylabel('Counts')
a1.set_xlabel('Phase');
a1.set_title('Angles at 20 Hz')

a2.set_title('Angles at 50 Hz')
a2.set_xlabel('Phase');

#avg ch of each layer


layer_avg= np.zeros((11,offset*2,K))

for trial in range(K):
    
    t=test[:,:,trial]
    
    means=[]
    count=0
    for c in range(11):
        t2=np.mean(t[0+count:11+count],axis=0)
        means.append(t2)
        count+=11

    layer_avg[:,:,trial]=means



ch_1 =layer_avg[0,:].T
ch_2 = layer_avg[1,:].T
T = 3 



j8 = where(f==20)[0][0]       # Determine index j for frequency 10 Hz.
j24 = where(f==30)[0][0]     # Determine index j for frequency 50 Hz.

phi8 = zeros(K)              # Variables to hold phase differences.
phi24 = zeros(K)

for k in range(K):           # For each trial, compute the cross spectrum. 
    x = ch_1[k] - mean(ch_1[k])  # Get the data from each electrode,
    y = ch_2[k] - mean(ch_2[k,:])
    xf = rfft(x - mean(x))   # ... compute the Fourier transform,
    yf = rfft(y - mean(y))
    Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    phi8[k] = angle(Sxy[j8]) # ... and the phases.
    phi24[k] = angle(Sxy[j24])
                             # Plot the distributions of phases.
_, (a1, a2) = subplots(1, 2, sharey=True, sharex=True)
a1.hist(phi8, bins=20, range=[-pi, pi])
a2.hist(phi24, bins=20, range=[-pi, pi])

#ylim([0, 40])                # Set y-axis and label axes.
a1.set_ylabel('Counts')
a1.set_xlabel('Phase');
a1.set_title('Angles at 30 Hz')

a2.set_title('Angles at 80 Hz')
a2.set_xlabel('Phase');


#avg shunks


shank_avg= np.zeros((11,offset*2,K))

for trial in range(K):
    
    t=test[:,:,trial]
    
    means=[]
    start_array = np.arange(0,121,11)
    for c in range(11):
        
        t2=np.mean(t[start_array],axis=0)
        means.append(t2)
        start_array+=1

    shank_avg[:,:,trial]=means



ch_1 =shank_avg[0,:].T
ch_2 = shank_avg[5,:].T
T = 3 



j8 = where(f==25)[0][0]       # Determine index j for frequency 10 Hz.
j24 = where(f==35)[0][0]     # Determine index j for frequency 50 Hz.

phi8 = zeros(K)              # Variables to hold phase differences.
phi24 = zeros(K)

for k in range(K):           # For each trial, compute the cross spectrum. 
    x = ch_1[k] - mean(ch_1[k])  # Get the data from each electrode,
    y = ch_2[k] - mean(ch_2[k,:])
    xf = rfft(x - mean(x))   # ... compute the Fourier transform,
    yf = rfft(y - mean(y))
    Sxy = 2 * dt**2 / T * (xf * conj(yf))  # ... and the cross-spectrum,
    phi8[k] = angle(Sxy[j8]) # ... and the phases.
    phi24[k] = angle(Sxy[j24])
                             # Plot the distributions of phases.
_, (a1, a2) = subplots(1, 2, sharey=True, sharex=True)
a1.hist(phi8, bins=20, range=[-pi, pi])
a2.hist(phi24, bins=20, range=[-pi, pi])

#ylim([0, 40])                # Set y-axis and label axes.
a1.set_ylabel('Counts')
a1.set_xlabel('Phase');
a1.set_title('Angles at 30 Hz')

a2.set_title('Angles at 80 Hz')
a2.set_xlabel('Phase');



