# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:30:10 2020

@author: KAMPFF-LAB-ANALYSIS3
"""











#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz    

probe_map_flatten = ephys.probe_map.flatten()



RAT_ID = RAT_ID_ephys

rat_summary_table_path=rat_summary_ephy


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


        #downsampled amplifier data opening 
        data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
        down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
        num_samples = int(int(len(down))/num_raw_channels)
        reshaped_down=  np.reshape(down,(num_samples,num_raw_channels))  
        down=None
        down_T = reshaped_down.T

        freq = 30000
        offset = 1500
        num_raw_channels = 128
        

        #remove the first early trials from event of choice 
        downsampled_event_idx = downsampled_end[1:]
        
        event_name= 'reward.csv'
         
       

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
          
        
        
        #finding chunks 
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
    










































#plot post event/ pre event from saved files 

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
    
    























