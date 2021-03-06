# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:13:11 2020

@author: KAMPFF-LAB-ANALYSIS3
"""
import os
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from scipy import stats 
import seaborn as sns
import glob


#delta = 1-4 Hz
#theta = 4-8 Hz
#alpha = 8-12 Hz
#beta = 12-30 Hz   


#test George cluster code 
hardrive_path = r'F:/'

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

            
summary_folder = 'F:/Videogame_Assay/LFP_summary/'            
event_folder =  'ball_on/'       

    
band = ['delta','theta','beta','alpha','gamma']
       

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
    
        shape = np.shape(sum_before)
        
        #reshaping to match fx requirements 11*11*trials
        
        matrix_before = np.zeros((11,11,shape[1]))
        matrix_after = np.zeros((11,11,shape[1]))
        
        for i in range(shape[1]):
            
            trial_before = np.reshape(sum_before[:,i],np.shape(probe_map)) #make it 11x11
            matrix_before[:,:,i] = trial_before # stacking them trial by trial
        
            trial_after = np.reshape(sum_after[:,i],np.shape(probe_map))
            matrix_after[:,:,i] = trial_after
        
        
        
        num_permutations=1000
        min_area = 3 
        
        p_values_sum, cluster_labels_under_alpha_sum = monte_carlo_significance_probability(matrix_before, matrix_after, 
                                                       num_permutations=num_permutations, min_area=min_area, cluster_alpha=0.01,
                                                       monte_carlo_alpha=0.05, sample_statistic='independent',
                                                       cluster_statistic='maxsum')
        
        p_values_area, cluster_labels_under_alpha_area = monte_carlo_significance_probability(matrix_before, matrix_after, 
                                                         num_permutations=num_permutations, min_area=min_area, cluster_alpha=0.01,
                                                         monte_carlo_alpha=0.05, sample_statistic='independent',
                                                         cluster_statistic='maxarea')
       
        
        csv_name_sum =  RAT_ID[r] + '_'+ band[b] +  '_clusters_maxsum_'+ str(min_area) + '.csv'
        csv_name_sum_p_value = RAT_ID[r] +  '_'+band[b] +  '_maxsum_p_values_'+ str(min_area) + '.csv'
        csv_name_area =  RAT_ID[r] + '_'+ band[b] +  '_clusters_maxerea_'+ str(min_area) + '.csv'
        csv_name_area_p_values =  RAT_ID[r] +  '_'+band[b] +  '_maxerea_p_values_'+ str(min_area) + '.csv'
  
        #savings
        np.savetxt(csv_dir_path + csv_name_sum , cluster_labels_under_alpha_sum, delimiter=',', fmt='%s')
        np.savetxt(csv_dir_path + csv_name_sum_p_value , p_values_sum, delimiter=',', fmt='%s')
        
        np.savetxt(csv_dir_path + csv_name_area , cluster_labels_under_alpha_area ,delimiter=',', fmt='%s')
        np.savetxt(csv_dir_path + csv_name_area_p_values , p_values_area, delimiter=',', fmt='%s')
        print(r) 
    print(b)


#plotting  scatter 

summary_folder = 'F:/Videogame_Assay/LFP_summary/'            
event_folder =  'ball_on/'       

    
band = ['delta','theta','beta','alpha']
       

for b in range(len(band)):
            
    for r in range(len(rat_summary_table_path)): 
        
        
        #retrieve bad channel frim the last session of each rat
        Level_2_post = prs.Level_2_post_paths(rat_summary_table_path[r])
        sessions_subset = Level_2_post   
        
        session = sessions_subset[-1]
        session_path =  os.path.join(hardrive_path,session) 
        csv_bad_ch = os.path.join(session_path +'/bad_channels.csv')
        bad_ch = np.genfromtxt(csv_bad_ch, delimiter = ',', dtype=int)  
        
        #session_path =  os.path.join(hardrive_path,session)    
        csv_dir_path = os.path.join(summary_folder + event_folder)
        offset_folder = 'pre/'
        csv_to_path = os.path.join(csv_dir_path + offset_folder)
           
        matching_files_before  = np.array(glob.glob(csv_to_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" ))
        sum_before = np.genfromtxt(matching_files_before[0], delimiter= ',',dtype= float)       
        
        offset_folder = 'post/'
        csv_to_path = os.path.join(csv_dir_path + offset_folder)
        
        matching_files_after = np.array(glob.glob(csv_to_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" ))
        sum_after = np.genfromtxt(matching_files_after[0], delimiter= ',',dtype= float)
        
        
        offset_folder = 'clusters_maxerea'
        matching_files = np.array(glob.glob(csv_dir_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" + "*"+offset_folder+"*" ))
        max_area =  np.genfromtxt(matching_files[0], delimiter= ',',dtype= float)
        
        offset_folder = 'clusters_maxsum'
        matching_files = np.array(glob.glob(csv_dir_path + "*"+RAT_ID[r]+"*"+ "*"+band[b]+"*" + "*"+offset_folder+"*" ))
        max_sum =  np.genfromtxt(matching_files[0], delimiter= ',',dtype= float)
        
        shape = np.shape(sum_before)
        
        matrix_before = np.zeros((11,11,shape[1]))
        matrix_after = np.zeros((11,11,shape[1]))
        
        for i in range(shape[1]):
            
            trial_before = np.reshape(sum_before[:,i],np.shape(probe_map))
            matrix_before[:,:,i] = trial_before
        
            trial_after = np.reshape(sum_after[:,i],np.shape(probe_map))
            matrix_after[:,:,i] = trial_after        
        
        
        #calculate norm diff or ratio between pre and post event 
        normalised_diff = (matrix_after-matrix_before)/matrix_before
        avg = np.mean(normalised_diff,axis=2)
        
        #ratio = matrix_after/matrix_before
        #avg_ratio = np.mean(ratio,axis=2)
        
        #need to be reshaped from 11x11 to 121 1D array
        #avg_ratio_to_plot = np.reshape(avg_ratio,121)
        avg_to_plot = np.reshape(avg,121)
        
        #probe like coordinates
        x,y= plotting_probe_coordinates()         
                       
        clusters_reshaped_sum = np.reshape(max_sum,121)
        #idx of the clusters, find where number is diff than 1
        cluster_indexes = [i for i,x in enumerate(clusters_reshaped_sum) if x != -1]
        #findt the remaining idx       
        no_indexes =[ele for ele in range(121) if ele not in cluster_indexes]
        
        #finds how many number different than -1 there are in the cluster mask,they correspond to  the number of groups of elevtrode
        #ideally would  like to color code them as scatter +
        clusters = np.delete(np.unique(max_sum), np.argwhere(np.unique(max_sum) == -1))
        
        
        #plot diff v  0-2 / plot ratio .5/1.5
        
        
        if cluster_indexes == []:
            
            #scatter probe
            f =plt.figure(figsize=(20,10))
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine(left=False)
            title = RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post'+ event_folder[:-1]
            figure_name =  RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post'+ event_folder[:-1] +'.png'
         
            plot=plt.scatter(x[no_indexes],np.array(y)[no_indexes], c = np.array(avg_to_plot)[no_indexes],cmap ="bwr", s=75, vmin=-.75,vmax=.75)
            #plt.scatter(x[no_indexes],np.array(y)[no_indexes], c='grey', alpha = .4,s=85, linewidth=.2)  
            plt.scatter(x[bad_ch],np.array(y)[bad_ch], c = 'w',s=85,marker='D')
            plt.scatter(x[bad_ch],np.array(y)[bad_ch], c = 'k',s=40,  edgecolors="k", linewidth=1.5, marker ='x')
            f.colorbar(plot)
            
            #plt.clim(0.0,2.0)
            plt.ylim(-500,5000)
            plt.hlines(4808,0,12)  
            plt.title(title)
            
            f.savefig('F:/Videogame_Assay/LFP_summary_plots/' + figure_name, transparent=False)
            plt.close()            
            
            #plot image with + where cluster    
            
            
#            f =plt.figure(figsize=(20,10))
#            sns.set()
#            sns.set_style('white')
#            sns.axes_style('white')
#            sns.despine(left=False)
#            #plt.title(title)
#            title = RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post_'+ event_folder[:-1]
#            figure_name =  RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post_image'+ event_folder[:-1] +'.png'
#            
#            
#            
#            interpolation = 'bilinear'
#            colormap='viridis'
#            #f=plt.figure()
#            ax = f.add_subplot(111)
#            im = ax.imshow(avg, aspect='auto',              
#                   interpolation=interpolation,
#                  cmap=colormap,origin='upper',vmin =- .75, vmax=.75) #vmin = 0.5, vmax=1.5
#
#            plt.colorbar(im, ax=ax, cmap=colormap)
#           
#            plt.title(title)
#            f.savefig('F:/Videogame_Assay/LFP_summary_plots/' + figure_name, transparent=False)
#            plt.close()
            
        else:
 
        
                #scatter probe
                f =plt.figure(figsize=(20,10))
                sns.set()
                sns.set_style('white')
                sns.axes_style('white')
                sns.despine(left=False)
                title = RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post'+ event_folder[:-1]
                figure_name =  RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post'+ event_folder[:-1] +'.png'
            
              
                
                import itertools
                markers = itertools.cycle(('D', 's', 'd', '^', 'p'))           
            
                for clust in np.arange(len(clusters)):

                    find_cluster = cluster_indexes = [i for i,x in enumerate(clusters_reshaped_sum) if x == clusters[clust]]
                    no_indexes =[ele for ele in range(121) if ele not in find_cluster]
                    
                    if r==0:
                        try:
                                       
                            find_cluster.remove(bad_ch) 
                            no_indexes.remove(bad_ch)
                            
                        except Exception:
                            continue
                            
                    else:
                        
                            
                        find_cluster = list(set(find_cluster) - set(bad_ch)) 
                        
                        no_indexes= list(set(no_indexes) - set(bad_ch))
                                    
                    
                    plot=plt.scatter(x[find_cluster],np.array(y)[find_cluster], c = np.array(avg_to_plot)[find_cluster], cmap ="bwr",s=75, edgecolors="k",linewidth=.6,marker= next(markers),vmin=-.75,vmax=.75)
                    plt.scatter(x[no_indexes],np.array(y)[no_indexes], c = np.array(avg_to_plot)[no_indexes],cmap ="bwr", s=40, linewidth=.2,vmin=-.75,vmax=.75)
                    plt.scatter(x[bad_ch],np.array(y)[bad_ch], c = 'w',s=85,marker='D')
                    plt.scatter(x[bad_ch],np.array(y)[bad_ch], c = 'k',s=75,  edgecolors="k", linewidth=1.5, marker ='x')  
                    
                f.colorbar(plot)
                    #plt.clim(-.75,.75)               
                plt.ylim(-500,5000)
                plt.hlines(4808,0,12)  
                plt.title(title)
                                
            
            
            
                f.savefig('F:/Videogame_Assay/LFP_summary_plots/' + figure_name, transparent=False)
                plt.close()
                
                
                #plot image with + where cluster    
                               
    #            
    #            f =plt.figure(figsize=(20,10))
    #            sns.set()
    #            sns.set_style('white')
    #            sns.axes_style('white')
    #            sns.despine(left=False)
    #            #plt.title(title)
    #            title = RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post_'+ event_folder[:-1]
    #            figure_name =  RAT_ID[r]+'_'+band[b]+ '_norm_diff_pre_post_image'+ event_folder[:-1] +'.png'
    #            
    #            
    #            
    #            interpolation = 'bilinear'
    #            colormap='viridis'
    #            #f=plt.figure()
    #            ax = f.add_subplot(111)
    #            im = ax.imshow(avg, aspect='auto',              
    #                   interpolation=interpolation,
    #                  cmap=colormap,origin='upper',vmin = -.75, vmax=.75)
    #    
    #            
    #            # create x and y for matching the scatter to the image 
    #            x_image =  np.array(list(np.arange(0,11))*11)
    #            y_list=[10,9,8,7,6,5,4,3,2,1,0]
    #            y_image = np.repeat(y_list,11)
    #            y_flipped_image = abs(y_image-10)
    #        
    #            ax.scatter(x_image[cluster_indexes],np.array(y_flipped_image)[cluster_indexes], color='w', marker='+', linewidth=2)
#    #            plt.colorbar(im, ax=ax, cmap=colormap)
#               
#                plt.title(title)
#                f.savefig('F:/Videogame_Assay/LFP_summary_plots/' + figure_name, transparent=False)
#                plt.close()
#                
        
        
        
    

        
        