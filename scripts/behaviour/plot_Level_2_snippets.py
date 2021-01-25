import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats




##idx for touching light
#x_touch = trial_table[:,6]
#y_touch = trial_table[:,7]
#x_pre = trial_table[:,8]
#y_pre = trial_table[:,9]
#x_ball = trial_table[:,16]
#y_ball = trial_table[:,17]

#idx for moving light
#
#x_touch = trial_table[:,8]
#y_touch = trial_table[:,9]
#x_pre = trial_table[:,10]
#y_pre = trial_table[:,11]
#x_ball = trial_table[:,24]
#y_ball = trial_table[:,25]

 all_trials = np.ones(num_trials, np.bool)
    
    # Compute speed around touch (all trials)
    filtered_trials = all_trials
    dx = np.diff(x_snippets[filtered_trials], axis=1)
    dy = np.diff(y_snippets[filtered_trials], axis=1)
    #avg_speed = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
    
    speed = np.sqrt(dx*dx + dy*dy)   
    # Measure noise and set threshold (however, this could be hard coded, 0.1 or so)
    
    noise = np.nanstd(speed)
    threshold = noise*5
    #threshold = 0.1
    
    # Find over threshold values
    over_indices = speed > noise
    
    # Set these to NaN
    speed[over_indices] = np.nan
    
    # Measure median/mean

    median_speed = np.nanmedian(speed, axis=0)
    mean_speed = np.nanmean(speed, axis=0)
    #smooth_mean = scipy.signal.savgol_filter(mean_speed, window, poly)
    #smooth_median = scipy.signal.savgol_filter(median_speed, window, poly)
    sem_avg_speed = stats.sem(speed, nan_policy='omit', axis=0)
  
        
##############################touching light 

#touching light spped plot

# Load trial table
touch_pre = 'F:/Videogame_Assay/Trial_tables/Trial_table_touching_light_pre_shaders/Trial_table_final_level_2_touching_light_pre_shaders.csv'
touch_ephys ='F:/Videogame_Assay/Trial_tables/Trial_table_touching_light_ephys_shaders/Trial_table_final_level_2_touching_light_ephys_shaders.csv'


touch_x = 'F:/Videogame_Assay/Snippets/snippets_around_touch/x_shaders_snippets_around_touch_all_rats_touching_light_pre.csv'
touch_y = 'F:/Videogame_Assay/Snippets/snippets_around_touch/y_shaders_snippets_around_touch_all_rats_touching_light_pre.csv'

touch_x_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_touch/x_shaders_snippets_around_touch_all_rats_touching_light_ephys.csv'
touch_y_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_touch/y_shaders_snippets_around_touch_all_rats_touching_light_ephys.csv'


Level_2_trial_tables = [touch_pre,touch_ephys]
Level_2_snippets = [[touch_x,touch_y],[touch_x_ephys,touch_y_ephys]]
plot_name = ['level_2_speed_around_touch_PRE_surgery_','level_2_speed_around_touch_EPHYS_']
results_dir = 'F:/Videogame_Assay/Snippets/snippets_plots/'

for t, table in enumerate(Level_2_trial_tables):
    
    
    trial_table = np.genfromtxt(table, delimiter=',')
    
    snippets = Level_2_snippets[t]
    x = snippets[0]  
    x_snippets  = np.genfromtxt(x, delimiter=',')
    
    y = snippets[1]
    y_snippets  = np.genfromtxt(y, delimiter=',')
    
    plot = plot_name[t]
    
    #idx of interest for plotting from the trial table

    x_touch = trial_table[:,6]
    y_touch = trial_table[:,7]
    x_pre = trial_table[:,8]
    y_pre = trial_table[:,9]
    x_ball = trial_table[:,16]
    y_ball = trial_table[:,17]
    
    
    
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
    
    late = [300,50]   
    
    from_above_trials = (y_touch > y_ball)
    from_below_trials = (y_touch < y_ball)
    early_trials = trial_counts < 50
    late_trials = trial_counts > late[t]
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
    threshold = noise*2
    #threshold = 0.1
    
    # Find over threshold values
    over_indices = speed > threshold
    
    # Set these to NaN
    speed[over_indices] = np.nan
    median_speed = np.nanmedian(speed, axis=0)
    mean_speed = np.nanmean(speed, axis=0)
    sem_avg_speed = stats.sem(speed, nan_policy='omit', axis=0)
        
    avg = [median_speed,mean_speed]
    sem = [sem_avg_speed,sem_avg_speed]
    avg_str = ['median','mean']
    
    for a in np.arange(len(avg)):
        
        avg_sel = avg[a]
        sem_sel = sem[a]
        agv_str_sel = avg_str[a]
    
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
                
                # Measure noise and set threshold (however, this could be hard coded, 0.1 or so)
                
                noise = np.nanstd(speed_cond_1)
                threshold = noise*2
                #threshold = 0.1
                
                # Find over threshold values
                over_indices = speed_cond_1 > threshold
                
                # Set these to NaN
                speed_cond_1[over_indices] = np.nan
                sem_cond_1 = stats.sem(speed_cond_1, nan_policy='omit', axis=0)
                
                # Compute speed around touch (from below trials)
                filtered_trials = condition_2
                dx = np.diff(x_snippets[filtered_trials], axis=1)
                dy = np.diff(y_snippets[filtered_trials], axis=1)
                speed_cond_2 = np.sqrt(dx*dx + dy*dy)
                noise = np.nanstd(speed_cond_2)
                threshold = noise*2
                #threshold = 0.1
                
                # Find over threshold values
                over_indices = speed_cond_2 > threshold
                
                # Set these to NaN
                speed_cond_2[over_indices] = np.nan
                sem_cond_2 = stats.sem(speed_cond_2, nan_policy='omit', axis=0)        
            
                if a==0 :
                    
            
                    median_speed_cond_1 = np.nanmedian(speed_cond_1, axis=0)
                    median_speed_cond_2 = np.nanmedian(speed_cond_2, axis=0) 
                    
    
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
                    plt.vlines(359, 0.0, max(avg_sel)*2, 'k')
                                
                    
                    plt.plot(avg_sel,color= 'k')
                    plt.fill_between(range(len(avg_sel)),avg_sel-sem_sel,avg_sel+sem_sel, alpha = alpha, facecolor ='k')
                    
                    plt.plot(median_speed_cond_1,color= color_1)
                    plt.fill_between(range(len(median_speed_cond_1)),median_speed_cond_1-sem_cond_1,median_speed_cond_1+sem_cond_1, alpha = alpha, facecolor = color_1)
                    
                    plt.plot(median_speed_cond_2,color= color_2)
                    plt.fill_between(range(len(median_speed_cond_2)),median_speed_cond_2-sem_cond_2,median_speed_cond_2+sem_cond_2, alpha = alpha, facecolor = color_2)
                    
                    ax.axes.get_yaxis().set_visible(True) 
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    plt.yticks(fontsize=15)
                    plt.xticks(fontsize=15)
                    plt.xlim((-50,800))
                    plt.ylim((0,0.005))
        
                    plt.title(agv_str_sel + '_'+string_1+'_'+color_1 +'_VS_'+ string_2+ '_'+color_2 )
                    plt.ylabel(agv_str_sel + '_speed_sem_shaders')
                    #plt.show()
                    
                    f.tight_layout()
                    
                    f.savefig(results_dir + figure_name, transparent=False)
                        
                        
                    
                else:
                  
                    
                   
                    mean_speed_cond_1 = np.nanmean(speed_cond_1, axis=0)           
                    mean_speed_cond_2 = np.nanmean(speed_cond_2, axis=0)
    
    
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
                    plt.vlines(359, 0.0, max(avg_sel)*2, 'k')
                                
                    
                    plt.plot(avg_sel,color= 'k')
                    plt.fill_between(range(len(avg_sel)),avg_sel-sem_sel,avg_sel+sem_sel, alpha = alpha, facecolor ='k')
                    
                    plt.plot(mean_speed_cond_1,color= color_1)
                    plt.fill_between(range(len(mean_speed_cond_1)),mean_speed_cond_1-sem_cond_1,mean_speed_cond_1+sem_cond_1, alpha = alpha, facecolor = color_1)
                    
                    plt.plot(mean_speed_cond_2,color= color_2)
                    plt.fill_between(range(len(mean_speed_cond_2)),mean_speed_cond_2-sem_cond_2,mean_speed_cond_2+sem_cond_2, alpha = alpha, facecolor = color_2)
                    
                    ax.axes.get_yaxis().set_visible(True) 
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    plt.yticks(fontsize=15)
                    plt.xticks(fontsize=15)
                    plt.xlim((-50,800))
                    plt.ylim((0,0.005))
        
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

from_above_trials = (y_touch > y_ball)
from_below_trials = (y_touch < y_ball)
early_trials = trial_counts < 50
late_trials = trial_counts > 300
from_left_trials = (x_touch < x_ball)
from_right_trials = (x_touch > x_ball)
missed_trials = outcome == 'Missed'
rewarded_trials = outcome == 'Food'



filename_x  = touch_x
x_snippets  = np.genfromtxt(filename_x, delimiter=',')
#x_snippets = np.convolve(x_snippets_open[0,:], kernel)



filename_y = touch_y
y_snippets= np.genfromtxt(filename_y, delimiter=',')
filename = touch_pre
# Load trial table

trial_table = np.genfromtxt(filename, delimiter=',')

# create a count of the trial (same than idx = 0)
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


#idx of interest for plotting from the trial table 


x_touch = trial_table[:,6]
y_touch = trial_table[:,7]
x_pre = trial_table[:,8]
y_pre = trial_table[:,9]
x_ball = trial_table[:,16]
y_ball = trial_table[:,17]

trial_table_outcome  = np.genfromtxt(filename, delimiter=',', dtype=str)
outcome = trial_table_outcome[:,-1]



num_trials = len(x_ball)

#kernel = np.ones(720)/720


# Load X and Y shader snippet arrays

touch_x = 'F:/Videogame_Assay/Snippets/snippets_around_touch/x_shaders_snippets_around_touch_all_rats_touching_light_pre.csv'
touch_y = 'F:/Videogame_Assay/Snippets/snippets_around_touch/y_shaders_snippets_around_touch_all_rats_touching_light_pre.csv'

touch_x_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_touch/x_shaders_snippets_around_touch_all_rats_touching_light_ephys.csv'
touch_y_ephys = 'F:/Videogame_Assay/Snippets/snippets_around_touch/y_shaders_snippets_around_touch_all_rats_touching_light_ephys.csv'



filename_x  = touch_x
x_snippets  = np.genfromtxt(filename_x, delimiter=',')
#x_snippets = np.convolve(x_snippets_open[0,:], kernel)



filename_y = touch_y
y_snippets= np.genfromtxt(filename_y, delimiter=',')
#y_snippets = np.convolve(y_snippets_open[0,:], kernel)





# Filter trials
all_trials = np.ones(num_trials, np.bool)



central_trials = (x_ball > 600) * (x_ball < 1100) * (y_ball > 450) * (y_ball < 850)

from_above_trials = (y_touch > y_ball)
from_below_trials = (y_touch < y_ball)
early_trials = trial_counts < 50
late_trials = trial_counts > 300
from_left_trials = (x_touch < x_ball)
from_right_trials = (x_touch > x_ball)
missed_trials = outcome == 'Missed'
rewarded_trials = outcome == 'Food'



# Compute speed around touch (all trials)
filtered_trials = all_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed = np.sqrt(dx*dx + dy*dy)
median_speed = np.nanmedian(speed, axis=0)
mean_speed = np.nanmean(speed, axis=0)
sem_avg_speed = stats.sem(speed, nan_policy='omit', axis=0)


# Compute speed around touch (from above trials)
filtered_trials = from_above_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed_from_above = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed_above = np.sqrt(dx*dx + dy*dy)
median_speed_from_above = np.nanmedian(speed_above, axis=0)
mean_speed_from_above = np.nanmean(speed_above, axis=0)
sem_above = stats.sem(speed_above, nan_policy='omit', axis=0)

# Compute speed around touch (from below trials)
filtered_trials = from_below_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
speed_below = np.sqrt(dx*dx + dy*dy)
median_speed_from_below = np.nanmedian(speed_below, axis=0)
mean_speed_from_below = np.nanmean(speed_below, axis=0)
sem_below = stats.sem(speed_below, nan_policy='omit', axis=0)


#plot MEDIAN above / below


plot_name = 'touching_light_pre_median_EPHYS.png'

# Plot average speeds

figure_name = 'median_speed_above_below_EPHYS' + plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(median_speed)*1.5, 'k')



plt.plot(median_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),median_speed-sem_avg_speed,median_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')

plt.plot(median_speed_from_above,color= 'r')
plt.fill_between(range(len(median_speed_from_above)),median_speed_from_above-sem_above,median_speed_from_above+sem_above, alpha = 0.4, facecolor ='r')

plt.plot(median_speed_from_below,color= 'b')
plt.fill_between(range(len(median_speed_from_below)),median_speed_from_below-sem_below,median_speed_from_below+sem_below, alpha = 0.4, facecolor ='b')

ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim((-50,800))
plt.ylim((0,0.005))
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.title("MEDIAN SPEED EPHYS: Above (r) vs Below (b)")
plt.ylabel('median_speed_sem_shaders_EPHYS')
#plt.show()

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)



#plot MEAN above / below


plot_name = 'touching_light_pre_MEAN_EPHYS.png'

# Plot average speeds

figure_name = 'mean_speed_above_below_EPHYS' + plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(mean_speed)*2, 'k')



plt.plot(mean_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),mean_speed-sem_avg_speed,mean_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')

plt.plot(mean_speed_from_above,color= 'r')
plt.fill_between(range(len(mean_speed_from_above)),mean_speed_from_above-sem_above,mean_speed_from_above+sem_above, alpha = 0.4, facecolor ='r')

plt.plot(mean_speed_from_below,color= 'b')
plt.fill_between(range(len(mean_speed_from_below)),mean_speed_from_below-sem_below,mean_speed_from_below+sem_below, alpha = 0.4, facecolor ='b')

ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.title("MEAN SPEED : Above (r) vs Below (b)")
plt.ylabel('mean_speed_sem_shaders EPHYS')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)







######################################################################


# Plot average speeds
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
#plt.title("Avg Speed: Above (r) vs Below (b)")
#plt.show()



# Plot speed early vs late

# Compute speed around touch (early trials)
filtered_trials = early_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
early_speed = np.sqrt(dx*dx + dy*dy)
median_speed_early = np.nanmedian(early_speed, axis=0)
mean_speed_early = np.nanmean(early_speed, axis=0)
sem_early = stats.sem(early_speed, nan_policy='omit', axis=0)


# Compute speed around touch (late trials)
filtered_trials = late_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
late_speed = np.sqrt(dx*dx + dy*dy)
median_speed_late = np.nanmedian(late_speed, axis=0)
mean_speed_late = np.nanmean(late_speed, axis=0)
sem_late = stats.sem(late_speed, nan_policy='omit', axis=0)


# Plot average speeds

plot_name = 'touching_light_pre_MEDIAN.png'

figure_name = 'median_speed_early_late_' +plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.vlines(359, 0, max(median_speed)*1.5, 'k')
#plt.vlines(359, 0.0, 3.0, 'k')

plt.plot(median_speed,color= 'k')
plt.fill_between(range(len(median_speed)),median_speed-sem_avg_speed,median_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')



plt.plot(median_speed_early,color= 'g')
plt.fill_between(range(len(median_speed_early)),median_speed_early-sem_early,median_speed_early+sem_early, alpha = 0.4, facecolor ='g')

plt.plot(median_speed_late,color= 'r')
plt.fill_between(range(len(median_speed_late)),median_speed_late-sem_late,median_speed_late+sem_late, alpha = 0.4, facecolor ='r')


ax.axes.get_xaxis().set_visible(True) 
ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_early, 'r')
#plt.plot(avg_speed_late, 'b')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.title("median SPEED : Early (g) 50  vs late (m) 300")
plt.ylabel('median_speed_sem_shaders ')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)



######plt mean


plot_name = 'touching_light_pre_mean.png'

figure_name = 'mean_speed_early_late_' +plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.vlines(359, 0, max(median_speed)*1.5, 'k')
#plt.vlines(359, 0.0, 3.0, 'k')


plt.plot(mean_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),mean_speed-sem_avg_speed,mean_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(mean_speed_early,color= 'g')
plt.fill_between(range(len(median_speed_early)),mean_speed_early-sem_early,mean_speed_early+sem_early, alpha = 0.4, facecolor ='g')

plt.plot(mean_speed_late,color= 'r')
plt.fill_between(range(len(median_speed_late)),mean_speed_late-sem_late,mean_speed_late+sem_late, alpha = 0.4, facecolor ='r')


ax.axes.get_xaxis().set_visible(True) 
ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_early, 'r')
#plt.plot(avg_speed_late, 'b')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.title("mean SPEED : Early (g) 50  vs late (m) 300")
plt.ylabel('mean_speed_sem_shaders ')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)





## Plot all snippets centered on ball
#num_trials = 1000
#plt.figure()
#for i in range(num_trials):
#    plt.plot(x_snippets[i,:] - x_snippets[i,360], y_snippets[i,:] - y_snippets[i, 360], alpha=0.01)
#plt.show()


# Plot speed left vs right

# Compute speed around touch (left trials)
filtered_trials = from_left_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
left_speed = np.sqrt(dx*dx + dy*dy)
median_speed_left = np.nanmedian(left_speed, axis=0)
mean_speed_left = np.nanmean(left_speed, axis=0)
sem_left = stats.sem(left_speed, nan_policy='omit', axis=0)

len(dx)
# Compute speed around touch (right trials)
filtered_trials = from_right_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
right_speed = np.sqrt(dx*dx + dy*dy)
median_speed_right = np.nanmedian(right_speed, axis=0)
mean_speed_right = np.nanmean(right_speed, axis=0)
sem_right = stats.sem(right_speed, nan_policy='omit', axis=0)
len(dx)

# Plot average speeds

plot_name = 'touching_light_pre_median.png'
figure_name = 'median_speed_left_right_' +plot_name
    
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(median_speed)*1.5, 'k')

plt.plot(median_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),median_speed-sem_avg_speed,median_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(median_speed_left,color= 'g')
plt.fill_between(range(len(median_speed_left)),median_speed_left-sem_left,median_speed_left+sem_left, alpha = 0.4, facecolor ='g')

plt.plot(median_speed_right,color= 'b')
plt.fill_between(range(len(median_speed_left)),median_speed_right-sem_right,median_speed_right+sem_right, alpha = 0.4, facecolor ='b')


ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15) 
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_early, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))

plt.title("medain SPEED : left g  vs right b")
plt.ylabel('median_speed_sem_shaders ')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)


###mean


plot_name = 'touching_light_pre_mean.png'
figure_name = 'mean_speed_left_right_' +plot_name
    
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(median_speed)*1.5, 'k')


plt.plot(mean_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),mean_speed-sem_avg_speed,mean_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(mean_speed_right,color= 'g')
plt.fill_between(range(len(mean_speed_right)),mean_speed_right-sem_left,mean_speed_right+sem_left, alpha = 0.4, facecolor ='g')

plt.plot(mean_speed_right,color= 'b')
plt.fill_between(range(len(mean_speed_right)),mean_speed_right-sem_right,mean_speed_right+sem_right, alpha = 0.4, facecolor ='b')


ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15) 
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_early, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))

plt.title("mean SPEED : left g  vs right b")
plt.ylabel('mean_speed_sem_shaders ')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)




# missed vs rewarded


# Compute speed around touch (all trials)
filtered_trials = all_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed = np.sqrt(dx*dx + dy*dy)
median_speed = np.nanmedian(speed, axis=0)
mean_speed = np.nanmean(speed, axis=0)
sem_avg_speed = stats.sem(speed, nan_policy='omit', axis=0)



# Compute speed around touch (missed trials)
filtered_trials = missed_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
miss_speed = np.sqrt(dx*dx + dy*dy)
median_speed_miss = np.nanmedian(miss_speed, axis=0)
mean_speed_miss = np.nanmean(miss_speed, axis=0)
sem_miss = stats.sem(miss_speed, nan_policy='omit', axis=0)

# Compute speed around touch (rewarded trials)
filtered_trials = rewarded_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
reward_speed = np.sqrt(dx*dx + dy*dy)
median_speed_reward = np.nanmedian(reward_speed, axis=0)
mean_speed_reward = np.nanmean(reward_speed, axis=0)
sem_reward = stats.sem(reward_speed, nan_policy='omit', axis=0)




plot_name = 'touching_light_pre_median.png'
figure_name = 'median_speed_outcome_' +plot_name
    
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(median_speed)*1.5, 'k')

plt.plot(median_speed,color= 'k')
plt.fill_between(range(len(median_speed)),median_speed-sem_avg_speed,median_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(median_speed_miss,color= 'm')
plt.fill_between(range(len(median_speed_miss)),median_speed_miss-sem_miss,median_speed_miss+sem_miss, alpha = 0.4, facecolor ='m')

plt.plot(median_speed_reward,color= 'b')
plt.fill_between(range(len(median_speed_reward)),median_speed_reward-sem_reward,median_speed_reward+sem_reward, alpha = 0.4, facecolor ='b')


ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15) 
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_early, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))

plt.title("medain SPEED : miss m  vs rewarded b")
plt.ylabel('median_speed_sem_shaders ')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)


#mean


plot_name = 'touching_light_pre_mean.png'
figure_name = 'mean_speed_outcome_' +plot_name
    
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(median_speed)*1.5, 'k')

plt.plot(mean_speed,color= 'k')
plt.fill_between(range(len(mean_speed)),mean_speed-sem_avg_speed,mean_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(mean_speed_miss,color= 'm')
plt.fill_between(range(len(mean_speed_miss)),mean_speed_miss-sem_miss,mean_speed_miss+sem_miss, alpha = 0.4, facecolor ='m')

plt.plot(mean_speed_reward,color= 'b')
plt.fill_between(range(len(mean_speed_reward)),mean_speed_reward-sem_reward,mean_speed_reward+sem_reward, alpha = 0.4, facecolor ='b')


ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15) 
#plt.vlines(359, 0.001, 0.004, 'g')
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_early, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.xlim((-50,800))
plt.ylim((0,0.005))

plt.title("mean SPEED : miss m  vs rewarded b")
plt.ylabel('mean_speed_sem_shaders ')

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=False)



