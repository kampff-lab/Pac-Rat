# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:30:10 2020

@author: KAMPFF-LAB-ANALYSIS3
"""




#####not over trials



#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz    

probe_map_flatten = ephys.probe_map.flatten()



RAT_ID = RAT_ID_ephys[0]

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
        downsampled_event_idx = downsampled_touch[1:]
        
        event_name= 'touch.csv'
         
       

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
    







#plot post event/ pre event from saved files USED

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
    
    

#try to save heatmaps with p values USED 

summary_folder_plots = 'F:/Videogame_Assay/LFP_summary_plots/'
summary_folder = 'F:/Videogame_Assay/LFP_summary/'
event_folder =  'ball_on/'

        

lfp_band = ['alpha','delta','theta','beta']

for rat in range(len(RAT_ID)):

    for b, band in enumerate(lfp_band):
        
        
        csv_dir_path = os.path.join(summary_folder_plots + event_folder)

        #matching_files_stats  = glob.glob(csv_dir_path +  "*stats*"+"*"+ "*"+band+"*" +"*"+RAT_ID[rat]+"*" )
        matching_files_stats  = glob.glob(csv_dir_path +  "*1samp*"+"*"+ "*"+band+"*" +"*"+RAT_ID[rat]+"*" )
        stats =  np.genfromtxt(matching_files_stats[0], delimiter = ',', dtype = float) 
    
        offset_folder = 'pre/'
        csv_to_path = os.path.join(summary_folder + event_folder + offset_folder )
        
        matching_files_pre  = glob.glob(csv_to_path +"*"+RAT_ID[rat]+"*"+ "*"+band+"*" )
        pre_event =  np.genfromtxt(matching_files_pre[0], delimiter = ',', dtype = float) 
       
    
        offset_folder = 'post/'
        csv_to_path = os.path.join(summary_folder + event_folder + offset_folder )
        
        matching_files_post  = glob.glob(csv_to_path +"*"+RAT_ID[rat]+"*"+ "*"+band+"*" )
        post_event =  np.genfromtxt(matching_files_post[0], delimiter = ',', dtype = float) 
    
        tot_trials = np.shape(post_event)[1]
    
        trial_ratio = post_event/pre_event
        
        #pre_mean = np.mean(pre_event, axis=1)
        #post_mean = np.mean(post_event, axis=1)
              
            
        Level_2_post = prs.Level_2_post_paths(rat_summary_table_path[rat])
        sessions_subset = Level_2_post   
         
        
        session = sessions_subset[-1]
        session_path =  os.path.join(hardrive_path,session) 
        csv_bad_ch = os.path.join(session_path +'/bad_channels.csv')
        bad_ch = np.genfromtxt(csv_bad_ch, delimiter = ',', dtype=int)
        
        f =plt.figure(figsize=(20,10))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine(left=False)
        #plt.title(title)
        title = RAT_ID[rat]+'_'+band+ '_post/pre_then_mean'+ event_folder[:-1] +'_'+ str(tot_trials)
        figure_name = RAT_ID[rat]+ '_'+band+ '_mean_trial_ratio_ttest_1samp'+ event_folder[:-1]+'.png'
        

        
        #to_plot = post_mean/pre_mean
    
    
        to_plot = np.mean(trial_ratio,axis=1)  
        
        #scatter heatmap
        
        bonferroni_annotation = stats[:,1]

       
        indexes = [i for i,x in enumerate(bonferroni_annotation) if x == 1]
        
        if indexes == []:
            
            x,y= plotting_probe_coordinates() 
            plt.scatter(x,np.array(y), c = np.array(to_plot), cmap="bwr",s=75,vmin = 0.5, vmax=1.5, edgecolors="k", linewidth=.2)
            plt.colorbar()
            plt.clim(.5,1.5)
            plt.scatter(x[bad_ch],np.array(y)[bad_ch], c = 'k',s=75,  edgecolors="k", linewidth=.2)
            plt.ylim(-500,5000)
            plt.hlines(4808,0,12)    
            plt.title(title)
            
        else:
            
            no_indexes =[ele for ele in range(121) if ele not in indexes]
            #len(indexes)+len(no_indexes)
            
            x,y= plotting_probe_coordinates()
            
            plt.scatter(x[indexes],np.array(y)[indexes], c =np.array(to_plot)[indexes], cmap="bwr", vmin = 0.5, vmax=1.5, marker=(5, 2),s=55, linewidth=1,edgecolors="k")
            plt.colorbar()
            plt.clim(.5,1.5)
            plt.hlines(4808,0,12)
            plt.ylim(-500,5000)
            plt.scatter(x[no_indexes],np.array(y)[no_indexes], c = np.array(to_plot)[no_indexes], cmap="bwr",s=75,vmin = 0.5, vmax=1.5, edgecolors="k", linewidth=.2)       
            plt.scatter(x[bad_ch],np.array(y)[bad_ch], c = 'k',s=75,  edgecolors="k", linewidth=.2)
            plt.title(title)  
                
            
#        #heatmap 
#        c = np.array(bad_ch.astype(int).tolist())
#        
#        to_plot[c]= np.nan
#        
#        band_final = np.reshape(to_plot,newshape=probe_map.shape)
#        bonferroni_annotation = stats[:,1]
#        bonf = np.reshape(bonferroni_annotation,newshape=probe_map.shape)
#    
##        
#        ax = sns.heatmap(band_final,annot=bonf,  cmap="bwr", vmin = 0, vmax=2, edgecolors='white', linewidths=1,
#                          annot_kws={"size": 10}, cbar_kws = dict(use_gridspec=False,location="right"))#,norm=LogNorm() # "YlGnBu" RdBu
#        ax.patch.set(hatch='//', edgecolor='black')
#        bottom, top = ax.get_ylim()
#        ax.set_ylim(bottom + 0.5, top - 0.5)
#    
#        plt.title(title)
    
        
        f.savefig(summary_folder_plots + figure_name, transparent=False)
        plt.close()
        print('figure_saved' + band + RAT_ID[rat])
    


#############################attempt all session mne 
            
            
            
        
probe_map_flatten = ephys.probe_map.flatten()



RAT_ID = RAT_ID_ephys[2]

rat_summary_table_path=rat_summary_ephys[2]



for r, rat in enumerate(rat_summary_table_path): 
    
    
    #rat = rat_summary_table_path[0]
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
    
        #downsampled amplifier data opening 
        data_down= os.path.join(session_path +'/Amplifier_downsampled.bin')
        down =  np.memmap(data_down, dtype = np.uint16, mode = 'r')
        num_raw_channels =128
        num_samples = int(int(len(down))/num_raw_channels)
        reshaped_down=  np.reshape(down,(num_samples,num_raw_channels))  
        down=None
        down_T = reshaped_down.T

        freq = 30000
        offset = 6000

        
        idx = np.arange(120000,num_samples-120000,offset)        
        print(len(idx))

        delta_all_channels_mean = np.zeros((N,len(idx)))
        alpha_all_channels_mean = np.zeros((N,len(idx)))
        theta_all_channels_mean = np.zeros((N,len(idx)))
        beta_all_channels_mean = np.zeros((N,len(idx)))
        gamma_all_channels_mean = np.zeros((N,len(idx)))
        
        delta_all_channels_std = np.zeros((N,len(idx)))
        alpha_all_channels_std = np.zeros((N,len(idx)))
        theta_all_channels_std = np.zeros((N,len(idx)))
        beta_all_channels_std = np.zeros((N,len(idx)))
        gamma_all_channels_std = np.zeros((N,len(idx)))

        f0 =plt.figure(figsize=(20,20))
        sns.set()
        sns.set_style('white')
        sns.axes_style('white')
        sns.despine() 

        for ch, channel in enumerate(probe_map_flatten): 
           
            delta_mean_ch=[]
            alpha_mean_ch =[]
            theta_mean_ch=[]
            beta_mean_ch=[]
            gamma_mean_ch=[]
             
            delta_std_ch =[]
            alpha_std_ch=[]
            beta_std_ch=[]
            theta_std_ch=[]
            gamma_std_ch=[]

            ch_downsampled = down_T[channel,:]#should use channel
            #down_T=None
                    
            tot_chunk =  np.zeros((len(idx),offset))      
            
            for i,index in enumerate(idx):

                
                tot_chunk[i,:]= ch_downsampled[index : index+offset]
                
            #half size chunk double  bandwidth   
            ch_downsampled = None
                                
                   
            p, f = time_frequency.psd_array_multitaper(tot_chunk, sfreq= 1000, fmin = 1, fmax = 100, bandwidth = 2.5, n_jobs = 8)
    
            
            p_mean = np.mean(p, axis=0)
            c = stats.sem(p, axis = 0)

        
        
            ax = f0.add_subplot(11, 11, 1+ch, frameon=False)#all the probe is 11 11
            
           
            plt.plot(f, p_mean, color = '#1E90FF',alpha=0.3, label = 'touch', linewidth= 1)
            plt.fill_between(f, p_mean-p_mean, p_mean+p_mean,
                             alpha=0.4, edgecolor='#1E90FF', facecolor='#00BFFF')#,vmin=0.4, vmax =1.9) blue
    

            
            for c in np.arange(len(idx)):
                                        
                  
                                
                delta_ch = [i for i,v in enumerate(f) if 1<= v <=4 ]
                delta_sel = p[:][c][delta_ch]        
                delta_mean = np.mean(delta_sel)
                delta_std = np.std(delta_sel)
                delta_mean_ch.append(delta_mean)
                delta_std_ch.append(delta_std)
                
                
                theta_ch = [i for i,v in enumerate(f) if 4<= v <=8 ]
                theta_sel = p[:][c][theta_ch]
                theta_mean = np.mean(theta_sel)
                theta_std= np.std(theta_sel)
                theta_mean_ch.append(theta_mean)
                theta_std_ch.append(theta_std)
                
                alpha_ch = [i for i,v in enumerate(f) if 8<= v <=12 ]
                alpha_sel = p[:][c][alpha_ch]
                alpha_mean = np.mean(alpha_sel)   
                alpha_std=np.std(alpha_sel)
                alpha_mean_ch.append(alpha_mean)
                alpha_std_ch.append(alpha_std)
                
                beta_ch = [i for i,v in enumerate(f) if 12<= v <=30 ]
                beta_sel = p[:][c][beta_ch]             
                beta_mean= np.mean(beta_sel)
                beta_std=np.std(beta_sel)
                beta_mean_ch.append(beta_mean)
                beta_std_ch.append(beta_std)


                gamma_ch =  [i for i,v in enumerate(f) if  30<= v <=45 or 55<= v <=100 ]

                gamma_sel = p[:][c][gamma_ch]             
                gamma_mean= np.mean(gamma_sel)
                gamma_std=np.std(gamma_sel)
                gamma_mean_ch.append(gamma_mean)
                gamma_std_ch.append(gamma_std)
                
                
            delta_all_channels_mean[ch,:] =   delta_mean_ch     
            alpha_all_channels_mean[ch,:] =   alpha_mean_ch   
            beta_all_channels_mean[ch,:] =   beta_mean_ch   
            theta_all_channels_mean[ch,:] =   theta_mean_ch   
            gamma_all_channels_mean[ch,:] =   gamma_mean_ch   
                    
            delta_all_channels_std[ch,:] =   delta_std_ch     
            alpha_all_channels_std[ch,:] =   alpha_std_ch   
            beta_all_channels_std[ch,:] =   beta_std_ch   
            theta_all_channels_std[ch,:] =   theta_std_ch   
            gamma_all_channels_std[ch,:] =   gamma_std_ch  
            print(ch)
   
      
  

session_path =  os.path.join(hardrive_path,session) 
csv_bad_ch = os.path.join(session_path +'/bad_channels.csv')
bad_ch = np.genfromtxt(csv_bad_ch, delimiter = ',', dtype=int)
             

#### z score  value ch - mean ch /std ch
            
 #delta           
delta_mean_all_ch =  np.mean(delta_all_channels_mean, axis= 1)
delta_std_all_ch = np.std(delta_all_channels_mean, axis= 1)

delta_z_score = np.zeros((N,len(idx)))

for value in range(N):
    ch_selection = delta_all_channels_mean[value,:]
    z_score = (ch_selection-delta_mean_all_ch[value])/delta_std_all_ch[value]
    delta_z_score[value,:]= z_score
    
    
    
    
good_ch =[ele for ele in range(121) if ele not in bad_ch]
delta_good= delta_z_score[good_ch]
delta_z_score=delta_good



count = 0
f =plt.figure(figsize=(20,10))
for line in range(N):
    plt.plot(range(len(idx)),delta_z_score[line,:]+count)
    count= count+1
plt.title('zscore_delta_' +session + '_offset_'+str(offset))
fig_name = 'zscore_delta'
f.savefig('E:/thesis_figures/z_score_test/'+ fig_name +'.png')

count = 0
f =plt.figure(figsize=(20,10))
for line in range(N-len(bad_ch)):
    plt.plot(range(len(idx)),delta_z_score[line,:]+count)
    count= count+1
plt.title('zscore_delta_' +session + '_offset_'+str(offset))
fig_name = 'zscore_delta'
f.savefig('E:/thesis_figures/z_score_test/'+ fig_name +'.png')



#
#
#for value in range(N):
#    f=plt.figure(figsize=(10,5))
#    #ax = f0.add_subplot(11, 11, 1+value, frameon=False)#all the probe is 11 11
#    plt.plot(range(len(idx)),delta_z_score[value,:])
#    plt.ylim(-2,5)
#    f.savefig('E:/thesis_figures/z_score_test/'+str(value)+'.png')
#    plt.close()
#    



f =plt.figure(figsize=(20,10))
for line in range(N):
    plt.plot(delta_z_score[line,:])
plt.ylim(-2,5)
f.savefig('E:/thesis_figures/z_score_test/'+'overlap'+'.png')
plt.close()

#theta

theta_mean_all_ch =  np.mean(theta_all_channels_mean, axis= 1)
theta_std_all_ch = np.std(theta_all_channels_mean, axis= 1)

theta_z_score = np.zeros((N,len(idx)))

for value in range(N):
    ch_selection = theta_all_channels_mean[value,:]
    z_score = (ch_selection-theta_mean_all_ch[value])/theta_std_all_ch[value]
    theta_z_score[value,:]= z_score
    


theta_good= theta_z_score[good_ch]
theta_z_score=theta_good

count = 0
f =plt.figure(figsize=(20,10))
for line in range(N-len(bad_ch)):
    plt.plot(range(len(idx)),theta_z_score[line,:]+count)
    count= count+1
plt.title('zscore_theta_' +session + '_offset_'+str(offset))
fig_name = 'zscore_theta'
f.savefig('E:/thesis_figures/z_score_test/'+ fig_name +'.png')



#for value in range(N):
#    f=plt.figure(figsize=(10,5))
#    #ax = f0.add_subplot(11, 11, 1+value, frameon=False)#all the probe is 11 11
#    plt.plot(range(len(idx)),theta_z_score[value,:])
#    plt.ylim(-2,5)
#    f.savefig('E:/thesis_figures/z_score_test/'+str(value)+'.png')
#    plt.close()
#    



f =plt.figure(figsize=(20,10))
for line in range(N):
    plt.plot(theta_z_score[line,:])
plt.ylim(-2,5)
f.savefig('E:/thesis_figures/z_score_test/'+'overlap'+'.png')
plt.close()







#aplha

alpha_mean_all_ch =  np.mean(alpha_all_channels_mean, axis= 1)
alpha_std_all_ch = np.std(alpha_all_channels_mean, axis= 1)

alpha_z_score = np.zeros((N,len(idx)))

for value in range(N):
    ch_selection = alpha_all_channels_mean[value,:]
    z_score = (ch_selection-alpha_mean_all_ch[value])/alpha_std_all_ch[value]
    alpha_z_score[value,:]= z_score
    

alpha_good= alpha_z_score[good_ch]
alpha_z_score=alpha_good



count = 0
f =plt.figure(figsize=(20,10))
for line in range(N-len(bad_ch)):
    plt.plot(range(len(idx)),alpha_z_score[line,:]+count)
    count= count+1
plt.title('zscore_alpha_' +session + '_offset_'+str(offset))
fig_name = 'zscore_alpha'
f.savefig('E:/thesis_figures/z_score_test/'+ fig_name +'.png')




#
#for value in range(N):
#    f=plt.figure(figsize=(10,5))
#    #ax = f0.add_subplot(11, 11, 1+value, frameon=False)#all the probe is 11 11
#    plt.plot(range(len(idx)),alpha_z_score[value,:])
#    plt.ylim(-2,5)
#    f.savefig('E:/thesis_figures/z_score_test/'+str(value)+'.png')
#    plt.close()
#    



f =plt.figure(figsize=(20,10))
for line in range(N):
    plt.plot(alpha_z_score[line,:])
plt.ylim(-2,5)
f.savefig('E:/thesis_figures/z_score_test/'+'overlap'+'.png')
plt.close()


#beta

beta_mean_all_ch =  np.mean(beta_all_channels_mean, axis= 1)
beta_std_all_ch = np.std(beta_all_channels_mean, axis= 1)

beta_z_score = np.zeros((N,len(idx)))

for value in range(N):
    ch_selection = beta_all_channels_mean[value,:]
    z_score = (ch_selection-beta_mean_all_ch[value])/beta_std_all_ch[value]
    beta_z_score[value,:]= z_score
    
beta_good= beta_z_score[good_ch]
beta_z_score=beta_good

count = 0
f =plt.figure(figsize=(20,10))
for line in range(N-len(bad_ch)):
    plt.plot(range(len(idx)),beta_z_score[line,:]+count)
    count= count+1
plt.title('zscore_beta_' +session + '_offset_'+str(offset))
fig_name = 'zscore_beta'
f.savefig('E:/thesis_figures/z_score_test/'+ fig_name +'.png')




#
#for value in range(N):
#    f=plt.figure(figsize=(10,5))
#    #ax = f0.add_subplot(11, 11, 1+value, frameon=False)#all the probe is 11 11
#    plt.plot(range(len(idx)),beta_z_score[value,:])
#    plt.ylim(-2,5)
#    f.savefig('E:/thesis_figures/z_score_test/'+str(value)+'.png')
#    plt.close()
    



f =plt.figure(figsize=(20,10))
for line in range(N):
    plt.plot(beta_z_score[line,:])
plt.ylim(-2,5)
f.savefig('E:/thesis_figures/z_score_test/'+'overlap'+'.png')
plt.close()


###gamma

gamma_mean_all_ch =  np.mean(gamma_all_channels_mean, axis= 1)
gamma_std_all_ch = np.std(gamma_all_channels_mean, axis= 1)

gamma_z_score = np.zeros((N,len(idx)))

for value in range(N):
    ch_selection = gamma_all_channels_mean[value,:]
    z_score = (ch_selection-gamma_mean_all_ch[value])/gamma_std_all_ch[value]
    gamma_z_score[value,:]= z_score
    
gamma_good= gamma_z_score[good_ch]
gamma_z_score=gamma_good

count = 0
f =plt.figure(figsize=(20,10))
for line in range(N-len(bad_ch)):
    plt.plot(range(len(idx)),gamma_z_score[line,:]+count)
    count= count+1
plt.title('zscore_gamma_' +session + '_offset_'+str(offset))
fig_name = 'zscore_gamma'
f.savefig('E:/thesis_figures/z_score_test/'+ fig_name +'.png')



#
#
#for value in range(N):
#    f=plt.figure(figsize=(10,5))
#    #ax = f0.add_subplot(11, 11, 1+value, frameon=False)#all the probe is 11 11
#    plt.plot(range(len(idx)),gamma_z_score[value,:])
#    plt.ylim(-2,5)
#    f.savefig('E:/thesis_figures/z_score_test/'+str(value)+'.png')
#    plt.close()
#    



f =plt.figure(figsize=(20,10))
for line in range(N):
    plt.plot(gamma_z_score[line,:])
plt.ylim(-2,5)
f.savefig('E:/thesis_figures/z_score_test/'+'overlap'+'.png')
plt.close()

#K = tot_chunk.shape[0]                            # Get the number of trials.
#f, a = subplots(figsize=(6, 6))            # Make a square axis
#a.imshow(tot_chunk,                               #... and show the image,
#           extent=[0, 6000, K, 1],  # ... with meaningful axes,
#           aspect='auto')
######################tracking binning test 



shader_tracking_path = os.path.join(session_path + '/events/' +'Tracking.csv')
shader_tracking = np.genfromtxt(shader_tracking_path)
#crop_path  = os.path.join(session_path  +'/crop.csv')
#crop=np.genfromtxt(crop_path, delimiter =',')

#x=crop[:,0]
#y=crop[:,1]
#
#x= shader_tracking[:,0]
#y= shader_tracking[:,1]

offset = 120

idx = np.arange(14400,len(shader_tracking)-14400,offset) 

avg_bins_x =  []
avg_bins_y =  []    
            
for i,index in enumerate(idx):
              
    chunk= shader_tracking[index : index+offset]
    #chunk= crop[index : index+offset]
    x_avg_chunk= np.mean(chunk[:,0])
    y_avg_chunk= np.mean(chunk[:,1])
    
    avg_bins_x.append(x_avg_chunk)
    avg_bins_y.append(y_avg_chunk)
 
dst = np.sqrt((np.diff(avg_bins_x))**2 + (np.diff(avg_bins_y))**2)

plt.figure()
plt.plot(avg_bins_x,avg_bins_y)
plt.figure()
plt.plot(dst)



 