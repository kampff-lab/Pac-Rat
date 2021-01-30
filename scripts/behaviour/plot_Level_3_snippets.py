import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy import stats
import statsmodels.api as sm
from scipy.signal import savgol_filter
import seaborn as sns
# Filter trials


#def smoothing_speed(speed_file):
#    
#    smooth = np.zeros(( np.shape(speed_file)[0], np.shape(speed_file)[1]))
#    
#    for t, trial in enumerate(speed_file):
#        smooth_trial= scipy.signal.savgol_filter(trial, window, poly,mode='nearest')
#        smooth[t,:]=smooth_trial
#        
#    return smooth



#central_trials = (x_ball > 600) * (x_ball < 1100) * (y_ball > 450) * (y_ball < 850)
#central_trials = (x_ball > -.3) * (x_ball < .3) * (y_ball > -.3) * (y_ball < .3)

##test cleaning 
#
#dx = np.diff(x_snippets[filtered_trials], axis=1)
#dy = np.diff(y_snippets[filtered_trials], axis=1)
#   
#
#
#idx_x =np.zeros((np.shape(dx)[0],719))
#idx_y =np.zeros((np.shape(dx)[0],719))
#
#th= 0.01
#
#for t in np.arange(np.shape(dx)[0]):
#    
#    sel_x = dx[t]
#    sel_y = dy[t]
#    
#    dx_abs = abs(sel_x)
#    dy_abs = abs(sel_y)
#    
#    results_x=[]
#    results_y=[]
#    
#    for i in np.arange(719):
#        
#        if dx_abs[i]>th:
#            results_x.append(i)            
#            
#        if  dy_abs[i]>th:
#            
#            results_y.append(i)
#        else:            
#            continue
#    
#    sel_x[results_x]=np.nan
#    sel_y[results_y]=np.nan
#    
#        
#    idx_x[t]=sel_x
#    idx_y[t]=sel_y
#
#
#speed_cleaned = np.sqrt(idx_x*idx_x + idx_y*idx_y)
#median_speed_cleaned = np.nanmedian(speed_cleaned, axis=0)
#mean_speed_cleaned = np.nanmean(speed_cleaned, axis=0)


x_snippets  = np.genfromtxt(x_trigger_ephys, delimiter=',')
y_snippets  = np.genfromtxt(y_trigger_ephys, delimiter=',')

# Measure speed as usual
dx = np.diff(x_snippets, axis=1)
dy = np.diff(y_snippets, axis=1)
speed = np.sqrt(dx*dx + dy*dy)   


# Measure noise and set threshold (however, this could be hard coded, 0.1 or so)
noise = np.nanstd(speed)
threshold = noise*5
#threshold = 0.1

# Find over threshold values
over_indices = speed > threshold

# Set these to NaN
speed[over_indices] = np.nan

# Measure median/mean
median_speed_cleaned = np.nanmedian(speed, axis=0)
mean_speed_cleaned = np.nanmean(speed, axis=0)



plt.figure()
plt.plot(mean_speed_cleaned)
plt.figure()
plt.plot(median_speed_cleaned)

###########

x_snippets  = np.genfromtxt(x_trigger_ephys, delimiter=',')
y_snippets  = np.genfromtxt(y_trigger_ephys, delimiter=',')

# Measure speed as usual
dx = np.diff(x_snippets, axis=1)
dy = np.diff(y_snippets, axis=1)
speed = np.sqrt(dx*dx + dy*dy)   

x_snippets  = np.genfromtxt(x_trigger_ephys, delimiter=',')
y_snippets  = np.genfromtxt(y_trigger_ephys, delimiter=',')

# Measure speed as usual
dx = np.diff(x_snippets, axis=1)
dy = np.diff(y_snippets, axis=1)
speed_raw = np.sqrt(dx*dx + dy*dy) 

mean_speed = np.nanmean(speed_raw,axis=0)
median_speed = np.nanmedian(speed_raw,axis=0) 

plt.figure()
plt.plot(mean_speed)
plt.figure()
plt.plot(median_speed)
plt.show()


smooth= scipy.signal.savgol_filter(mean_speed_cleaned, 51, 3)
plt.plot(smooth)

 sem_avg_speed = stats.sem(smooth, nan_policy='omit', axis=0)

##############################moving light 
def sliding_window(speeds, window_size=6):    
    # Smooth speeds
    speeds_smoothed = np.copy(speeds)
    
    # Sliding window smoothing
    #window_size = 6 # Means +/- 6 samples from centre sample (~100 ms window)
    num_trials = speeds.shape[0]
    trial_length = speeds.shape[1]
    for i in range(num_trials):
        tmp_speed = np.copy(speeds[i][:])
        smooth_speed = np.copy(speeds[i][:])
        for s in range(window_size, trial_length-window_size):
            smooth_speed[s] = np.nanmean(tmp_speed[(s-window_size):(s+window_size+1)])
        speeds_smoothed[i] = smooth_speed
        print(i)
    return speeds_smoothed

# Load trial table
moving_pre = 'F:/Videogame_Assay/Trial_tables/Trial_table_moving_light_pre_shaders/Trial_table_final_level_3_moving_light_pre_shaders.csv'
moving_ephys ='F:/Videogame_Assay/Trial_tables/Trial_table_moving_light_ephys_shaders/Trial_table_final_level_3_moving_light_ephys_shaders.csv'


trigger_x = 'F:/Videogame_Assay/Snippets/snippets_around_trigger/x_shaders_snippets_around_touch_all_rats_moving_light_pre.csv'
trigger_y = 'F:/Videogame_Assay/Snippets/snippets_around_trigger/y_shaders_snippets_around_touch_all_rats_moving_light_pre.csv'

x_trigger_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_trigger/x_shaders_snippets_around_touch_all_rats_moving_light_ephys.csv'
y_trigger_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_trigger/y_shaders_snippets_around_touch_all_rats_moving_light_ephys.csv'

x_catch_ephys ='F:/Videogame_Assay/Snippets/snippets_around_catch/x_shaders_snippets_around_touch_all_rats_moving_light_ephys_catch.csv'
y_catch_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_catch/y_shaders_snippets_around_touch_all_rats_moving_light_ephys_catch.csv'

x_catch =  'F:/Videogame_Assay/Snippets/snippets_around_catch/x_shaders_snippets_around_touch_all_rats_moving_light_pre_catch.csv'
y_catch = 'F:/Videogame_Assay/Snippets/snippets_around_catch/y_shaders_snippets_around_touch_all_rats_moving_light_pre_catch.csv'



Level_3_trial_tables = [moving_pre,moving_ephys]

Level_3_snippets_trigger = [[trigger_x,trigger_y],
                            [x_trigger_ephys,y_trigger_ephys]]


Level_3_snippets_catch = [[x_catch,y_catch],
                  [x_catch_ephys,y_catch_ephys]]

plot_name_trigger = ['level_3_speed_trigger_around_touch_PRE_surgery_','level_3_speed_trigger_around_touch_EPHYS_']
plot_name_catch = ['level_3_speed_catch_around_touch_PRE_surgery_','level_3_speed_catch_around_touch_EPHYS_']

results_dir = 'F:/Videogame_Assay/Snippets/snippets_plots/'

window=15
poly =3

# choose which one to plot 
Level_3_snippets = Level_3_snippets_catch
plot_name = plot_name_catch
lim =0.007# 0.005 

#CHANGE IDX FROM TRIGGER TO TOUCH BEFORE RUNNING THE CODE 

for t, table in enumerate(Level_3_trial_tables):
    
    
    trial_table = np.genfromtxt(table, delimiter=',')
    
    snippets = Level_3_snippets[t]
    x = snippets[0]  
    x_snippets  = np.genfromtxt(x, delimiter=',')
    
    y = snippets[1]
    y_snippets  = np.genfromtxt(y, delimiter=',')
    
    plot = plot_name[t]
    
    #idx of interest for plotting from the trial table

    x_touch = trial_table[:,8]
    y_touch = trial_table[:,9]
    x_pre = trial_table[:,10]
    y_pre = trial_table[:,11]
    x_ball = trial_table[:,24]
    y_ball = trial_table[:,25]
    x_trigger = trial_table[:,14]
    y_trigger = trial_table[:,15]
        
    
    ids = trial_table[:,0]
    current_id = 0
    current_count = 0
    trial_counts = []
    for id in ids:
        if current_id != id:
            current_count = 0
            current_id = id
        else:
            current_count = current_count + 1
        trial_counts.append(current_count)
    trial_counts = np.array(trial_counts)
    
    print(len(trial_counts))

    trial_table_outcome  = np.genfromtxt(table, delimiter=',', dtype=str)
    outcome = trial_table_outcome[:,-1]    
    
    late = [300,100]   
    
    #from_above_trials = (y_trigger > y_ball)
    #from_below_trials = (y_trigger < y_ball)        
    from_above_trials = (y_touch > y_ball)
    from_below_trials = (y_touch < y_ball)
    
    early_trials = trial_counts < 50
    late_trials = trial_counts > late[t]
    
    #from_left_trials = (x_trigger < x_ball)
    #from_right_trials = (x_trigger > x_ball)
    from_left_trials = (x_touch < x_ball)
    from_right_trials = (x_touch > x_ball)
    
    missed_trials = outcome == 'Missed'
    rewarded_trials = outcome == 'Food'


    conditions = [[from_above_trials,from_below_trials],
                  [early_trials,late_trials],
                  [from_left_trials,from_right_trials],
                  [missed_trials,rewarded_trials]]

    conditions_str =  [['above_trials','below_trials'],
                  ['early_trials','late_trials'],
                  ['left_trials','right_trials'],
                  ['missed_trials','rewarded_trials']]

    condition_colors = [['b','r'],
                        ['g','r'],
                        ['b','g'],
                        ['m','b']]
    
    agv_str_sel= 'median'
    num_trials = len(x_ball)

    # Filter trials
    all_trials = np.ones(num_trials, np.bool)
    
    # Compute speed around touch (all trials)
    filtered_trials = all_trials
    dx = np.diff(x_snippets[filtered_trials], axis=1)
    dy = np.diff(y_snippets[filtered_trials], axis=1)
    #avg_speed = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
    
    speed = np.sqrt(dx*dx + dy*dy)   
    # Measure noise and set threshold (however, this could be hard coded, 0.1 or so)
    
    noise = np.nanstd(speed)
    threshold = noise*3
    #threshold = 0.1
    
    # Find over threshold values
    over_indices = speed > threshold
    
    # Set these to NaN
    speed[over_indices] = np.nan
    
    # Measure median/mean
    smooth_speed = sliding_window(speed, window_size=6)
    
    median_speed = np.nanmedian(smooth_speed, axis=0)   
    sem_avg_speed = stats.sem(smooth_speed, nan_policy='omit', axis=0)
  
   

    for c, con in enumerate(conditions):
        try:
        
        
            condition_1 = con[0]
            condition_2 = con[1]
            string_1 =conditions_str[c][0]
            string_2 = conditions_str[c][1]
            color_1 = condition_colors[c][0]
            color_2 = condition_colors[c][1]
 


            # Compute speed around touch (from above trials)
            filtered_trials = condition_1
            dx = np.diff(x_snippets[filtered_trials], axis=1)
            dy = np.diff(y_snippets[filtered_trials], axis=1)
            #avg_speed_from_above = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
            speed_cond_1 = np.sqrt(dx*dx + dy*dy)
            noise = np.nanstd(speed_cond_1)
            threshold = noise*3
            #threshold = 0.1
            
            # Find over threshold values
            over_indices = speed_cond_1 > threshold
            
            # Set these to NaN
            speed_cond_1[over_indices] = np.nan
            
            smooth_cond_1 =  sliding_window(speed_cond_1, window_size=6)
            
            median_speed_cond_1 = np.nanmedian(smooth_cond_1, axis=0)
            sem_cond_1 = stats.sem(smooth_cond_1, nan_policy='omit', axis=0) 
            
            # Compute speed around touch (from below trials)
            filtered_trials = condition_2
            dx = np.diff(x_snippets[filtered_trials], axis=1)
            dy = np.diff(y_snippets[filtered_trials], axis=1)
            speed_cond_2 = np.sqrt(dx*dx + dy*dy)
            noise = np.nanstd(speed_cond_2)
            threshold = noise*3
            #threshold = 0.1
            
            # Find over threshold values
            over_indices = speed_cond_2 > threshold
            
            # Set these to NaN
            speed_cond_2[over_indices] = np.nan
            
            smooth_cond_2 =sliding_window(speed_cond_2, window_size=6)
            
            
            median_speed_cond_2 = np.nanmedian(smooth_cond_2, axis=0)
            sem_cond_2 = stats.sem(smooth_cond_2, nan_policy='omit', axis=0)        
        
            c1 = median_speed_cond_1
            c2 = median_speed_cond_2


            plot_condition =agv_str_sel+'_'+ string_1 +'_VS_' + string_2 +'.png'
            
            # Plot average speeds
            
            figure_name = plot + plot_condition
              
            f,ax = plt.subplots(figsize=(9,5))
            
            sns.set()
            sns.set_style('white')
            sns.axes_style('white')
            sns.despine(left=False)
            
            alpha = .4
            
            #plt.vlines(359, 0.001, 0.004, 'k')
            plt.vlines(359, 0.0, max(median_speed)*2, 'k')
                        
            
            plt.plot(median_speed,color= 'k')
            plt.fill_between(range(len(median_speed)),median_speed-sem_avg_speed,median_speed+sem_avg_speed, alpha = alpha, facecolor ='k')
            
            plt.plot(c1 ,color= color_1)
            plt.fill_between(range(len(c1 )),c1 -sem_cond_1,c1 +sem_cond_1, alpha = alpha, facecolor = color_1)
            
            plt.plot(c2,color= color_2)
            plt.fill_between(range(len(c2)),c2-sem_cond_2,c2+sem_cond_2, alpha = alpha, facecolor = color_2)
            
            ax.axes.get_yaxis().set_visible(True) 
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)
            plt.xlim((-50,800))
            plt.ylim((0,lim))

            plt.title(agv_str_sel + '_'+string_1+'_'+color_1 +'_VS_'+ string_2+ '_'+color_2)
            plt.ylabel(agv_str_sel + '_speed_sem_shaders')
            #plt.show()
            
            f.tight_layout()
            
            f.savefig(results_dir + figure_name, transparent=False)
            #print(con)
        except Exception:
            print(table + '_error')
            continue 
 
    print(table)







#
#
#
#
#
#
#
#
#
#
#
#        
#    #avg = [median_speed,mean_speed]
#    avg= [smooth_median,smooth_mean]
#    sem = [sem_avg_speed,sem_avg_speed]
#    avg_str = ['median','mean']
#    
#    for a in np.arange(len(avg)):
#        
#        avg_sel = avg[a]
#        sem_sel = sem[a]
#        agv_str_sel = avg_str[a]
#    
#        for c, con in enumerate(conditions):
#            try:
#            
#            
#                condition_1 = con[0]
#                condition_2 = con[1]
#                string_1 =conditions_str[c][0]
#                string_2 = conditions_str[c][1]
#                color_1 = condition_colors[c][0]
#                color_2 = condition_colors[c][1]
#     
#    
#    
#                # Compute speed around touch (from above trials)
#                filtered_trials = condition_1
#                dx = np.diff(x_snippets[filtered_trials], axis=1)
#                dy = np.diff(y_snippets[filtered_trials], axis=1)
#                #avg_speed_from_above = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
#                speed_cond_1 = np.sqrt(dx*dx + dy*dy)
#                noise = np.nanstd(speed_cond_1)
#                threshold = noise*2
#                #threshold = 0.1
#                
#                # Find over threshold values
#                over_indices = speed_cond_1 > threshold
#                
#                # Set these to NaN
#                speed_cond_1[over_indices] = np.nan
#                
#                sem_cond_1 = stats.sem(speed_cond_1, nan_policy='omit', axis=0)
#                
#                # Compute speed around touch (from below trials)
#                filtered_trials = condition_2
#                dx = np.diff(x_snippets[filtered_trials], axis=1)
#                dy = np.diff(y_snippets[filtered_trials], axis=1)
#                speed_cond_2 = np.sqrt(dx*dx + dy*dy)
#                noise = np.nanstd(speed_cond_2)
#                threshold = noise*2
#                #threshold = 0.1
#                
#                # Find over threshold values
#                over_indices = speed_cond_2 > threshold
#                
#                # Set these to NaN
#                speed_cond_2[over_indices] = np.nan
#                
#                sem_cond_2 = stats.sem(speed_cond_2, nan_policy='omit', axis=0)        
#            
#                if a==0 :
#                    
#            
#                    median_speed_cond_1 = np.nanmedian(speed_cond_1, axis=0)
#                    smooth_speed_cond_1 = scipy.signal.savgol_filter(median_speed_cond_1,window, poly)
#                    
#                    median_speed_cond_2 = np.nanmedian(speed_cond_2, axis=0) 
#                    smooth_speed_cond_2 = scipy.signal.savgol_filter(median_speed_cond_2,window, poly)
#                    
#                    c1 = smooth_speed_cond_1
#                    c2 = smooth_speed_cond_2
#                    
#    
#                    plot_condition =agv_str_sel+'_'+ string_1 +'_VS_' + string_2 +'.png'
#                    
#                    # Plot average speeds
#                    
#                    figure_name = plot + plot_condition
#                      
#                    f,ax = plt.subplots(figsize=(9,5))
#                    
#                    sns.set()
#                    sns.set_style('white')
#                    sns.axes_style('white')
#                    sns.despine(left=False)
#                    
#                    alpha = .4
#                    
#                    #plt.vlines(359, 0.001, 0.004, 'k')
#                    plt.vlines(359, 0.0, max(median_speed)*2, 'k')
#                                
#                    
#                    plt.plot(avg_sel,color= 'k')
#                    plt.fill_between(range(len(avg_sel)),avg_sel-sem_sel,avg_sel+sem_sel, alpha = alpha, facecolor ='k')
#                    
#                    plt.plot(c1,color= color_1)
#                    plt.fill_between(range(len(c1)),c1-sem_cond_1,c1+sem_cond_1, alpha = alpha, facecolor = color_1)
#                    
#                    plt.plot(c2,color= color_2)
#                    plt.fill_between(range(len(c2)),c2-sem_cond_2,c2+sem_cond_2, alpha = alpha, facecolor = color_2)
#                    
#                    ax.axes.get_yaxis().set_visible(True) 
#                    ax.yaxis.set_ticks_position('left')
#                    ax.xaxis.set_ticks_position('bottom')
#                    plt.yticks(fontsize=15)
#                    plt.xticks(fontsize=15)
#                    plt.xlim((-50,800))
#                    plt.ylim((0,lim))
#        
#                    plt.title(agv_str_sel + '_'+string_1+'_'+color_1 +'_VS_'+ string_2+ '_'+color_2 )
#                    plt.ylabel(agv_str_sel + '_speed_sem_shaders')
#                    #plt.show()
#                    
#                    f.tight_layout()
#                    
#                    f.savefig(results_dir + figure_name, transparent=False)
#                        
#                        
#                    
#                else:
#                  
#                    
#                   
#                    mean_speed_cond_1 = np.nanmean(speed_cond_1, axis=0)   
#                    smooth_speed_cond_1 = scipy.signal.savgol_filter(mean_speed_cond_1, window, poly)
#                    
#                    mean_speed_cond_2 = np.nanmean(speed_cond_2, axis=0)
#                    smooth_speed_cond_2 = scipy.signal.savgol_filter(mean_speed_cond_2, window, poly)
#                    
#                    c1 =smooth_speed_cond_1
#                    c2 = smooth_speed_cond_2
#    
#    
#                    plot_condition =agv_str_sel+'_'+ string_1 +'_VS_' + string_2 +'.png'
#                    
#                    # Plot average speeds
#                    
#                    figure_name = plot + plot_condition
#                      
#                    f,ax = plt.subplots(figsize=(9,5))
#                    
#                    sns.set()
#                    sns.set_style('white')
#                    sns.axes_style('white')
#                    sns.despine(left=False)
#                    
#                    alpha = .4
#                    
#                    #plt.vlines(359, 0.001, 0.004, 'k')
#                    plt.vlines(359, 0.0, max(median_speed)*2, 'k')
#                                
#                    
#                    plt.plot(avg_sel,color= 'k')
#                    plt.fill_between(range(len(avg_sel)),avg_sel-sem_sel,avg_sel+sem_sel, alpha = alpha, facecolor ='k')
#                    
#                    plt.plot(c1 ,color= color_1)
#                    plt.fill_between(range(len(c1 )),c1 -sem_cond_1,c1 +sem_cond_1, alpha = alpha, facecolor = color_1)
#                    
#                    plt.plot(c2,color= color_2)
#                    plt.fill_between(range(len(c2)),c2-sem_cond_2,c2+sem_cond_2, alpha = alpha, facecolor = color_2)
#                    
#                    ax.axes.get_yaxis().set_visible(True) 
#                    ax.yaxis.set_ticks_position('left')
#                    ax.xaxis.set_ticks_position('bottom')
#                    plt.yticks(fontsize=15)
#                    plt.xticks(fontsize=15)
#                    plt.xlim((-50,800))
#                    plt.ylim((0,lim))
#        
#                    plt.title(agv_str_sel + '_'+string_1+'_'+color_1 +'_VS_'+ string_2+ '_'+color_2)
#                    plt.ylabel(agv_str_sel + '_speed_sem_shaders')
#                    #plt.show()
#                    
#                    f.tight_layout()
#                    
#                    f.savefig(results_dir + figure_name, transparent=False)
#                    #print(con)
#            except Exception:
#                print(table + '_error')
#                continue 
#     
#    print(table)
#
#
#
#
#
