import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats





# Load trial table
filename = 'E:/thesis_figures/Tracking_figures/Trial_table_final_level_3_joystick.csv'
trial_table = np.genfromtxt(filename, delimiter=',')
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
#x_touch = trial_table[:,6]
#y_touch = trial_table[:,7]
#x_pre = trial_table[:,8]
#y_pre = trial_table[:,9]
#x_ball = trial_table[:,16]
#y_ball = trial_table[:,17]

x_touch = trial_table[:,8]
y_touch = trial_table[:,9]
x_pre = trial_table[:,10]
y_pre = trial_table[:,11]
x_ball = trial_table[:,24]
y_ball = trial_table[:,25]


num_trials = len(x_ball)


# Load X and Y shader snippet arrays
filename = 'E:/thesis_figures/Tracking_figures/x_crop_snippets_around_touch_joystick.csv'
x_snippets  = np.genfromtxt(filename, delimiter=',')

filename = 'E:/thesis_figures/Tracking_figures/y_crop_snippets_around_touch_joystick.csv'
y_snippets= np.genfromtxt(filename, delimiter=',')

# Filter trials
all_trials = np.ones(num_trials, np.bool)
#central_trials = (x_ball > 600) * (x_ball < 1100) * (y_ball > 450) * (y_ball < 850)
central_trials = (x_ball > -.3) * (x_ball < .3) * (y_ball > -.3) * (y_ball < .3)

from_above_trials = (y_touch > y_ball)
from_below_trials = (y_touch < y_ball)
early_trials = trial_counts < 50
late_trials = trial_counts > 300
from_left_trials = (x_touch < x_ball)
from_right_trials = (x_touch > x_ball)

# Compute speed around touch (all trials)
filtered_trials = all_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed = np.sqrt(dx*dx + dy*dy)
avg_speed = np.nanmedian(speed, axis=0)
sem_avg_speed = stats.sem(speed, nan_policy='omit', axis=0)


# Compute speed around touch (from above trials)
filtered_trials = from_above_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed_from_above = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed_above = np.sqrt(dx*dx + dy*dy)
avg_speed_from_above = np.nanmedian(speed_above, axis=0)
sem_above = stats.sem(speed_above, nan_policy='omit', axis=0)


filtered_trials = from_below_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
speed_below = np.sqrt(dx*dx + dy*dy)
avg_speed_from_below = np.nanmedian(speed_below, axis=0)
sem_below = stats.sem(speed_below, nan_policy='omit', axis=0)


plot_name = 'joystick.png'

# Plot average speeds

figure_name = 'median_speed_above_below_' +plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(avg_speed)+.5, 'k')



plt.plot(avg_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),avg_speed-sem_avg_speed,avg_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')

plt.plot(avg_speed_from_above,color= 'r')
plt.fill_between(range(len(avg_speed_from_above)),avg_speed_from_above-sem_above,avg_speed_from_above+sem_above, alpha = 0.4, facecolor ='r')

plt.plot(avg_speed_from_below,color= 'b')
plt.fill_between(range(len(avg_speed_from_below)),avg_speed_from_below-sem_below,avg_speed_from_below+sem_below, alpha = 0.4, facecolor ='b')

ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.title("Avg Speed : Above (r) vs Below (b)")
#plt.show()

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)




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
avg_speed_early = np.nanmedian(early_speed, axis=0)
sem_early = stats.sem(early_speed, nan_policy='omit', axis=0)


# Compute speed around touch (late trials)
filtered_trials = late_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
late_speed = np.sqrt(dx*dx + dy*dy)
avg_speed_late = np.nanmedian(late_speed, axis=0)
sem_late = stats.sem(late_speed, nan_policy='omit', axis=0)


# Plot average speeds

figure_name = 'median_speed_early_late_' +plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.vlines(359, 0, max(avg_speed)+.5, 'k')
#plt.vlines(359, 0.0, 3.0, 'k')

plt.plot(avg_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),avg_speed-sem_avg_speed,avg_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(avg_speed_early,color= 'r')
plt.fill_between(range(len(avg_speed_early)),avg_speed_early-sem_early,avg_speed_early+sem_early, alpha = 0.4, facecolor ='r')

plt.plot(avg_speed_late,color= 'b')
plt.fill_between(range(len(avg_speed_late)),avg_speed_late-sem_late,avg_speed_late+sem_late, alpha = 0.4, facecolor ='b')


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
plt.title("Avg Speed : Early (r) vs Late (b)")
plt.show()
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)




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
avg_speed_left = np.nanmedian(left_speed, axis=0)
sem_left = stats.sem(left_speed, nan_policy='omit', axis=0)

len(dx)
# Compute speed around touch (right trials)
filtered_trials = from_right_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
right_speed = np.sqrt(dx*dx + dy*dy)
avg_speed_right = np.nanmedian(right_speed, axis=0)
sem_right = stats.sem(right_speed, nan_policy='omit', axis=0)
len(dx)

# Plot average speeds


figure_name = 'median_speed_left_right_' +plot_name
    
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(avg_speed)+.5, 'k')

plt.plot(avg_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),avg_speed-sem_avg_speed,avg_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(avg_speed_left,color= 'r')
plt.fill_between(range(len(avg_speed_left)),avg_speed_left-sem_left,avg_speed_left+sem_left, alpha = 0.4, facecolor ='r')

plt.plot(avg_speed_right,color= 'b')
plt.fill_between(range(len(avg_speed_right)),avg_speed_right-sem_right,avg_speed_right+sem_right, alpha = 0.4, facecolor ='b')


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
plt.title("Avg Speed : left (r) vs right (b)")
plt.show()


f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)

##############################touching light 






# Load trial table
filename = 'F:/Videogame_Assay/Trial_tables/Trial_table_final_level_2_touching_light.csv'
trial_table = np.genfromtxt(filename, delimiter=',')
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

x_touch = trial_table[:,6]
y_touch = trial_table[:,7]
x_pre = trial_table[:,8]
y_pre = trial_table[:,9]
x_ball = trial_table[:,16]
y_ball = trial_table[:,17]



num_trials = len(x_ball)




kernel = np.ones(720)/720

x_smooth = np.convolve(x_centroid, kernel, mode ='same')
y_smooth = np.convolve(y_centroid, kernel, mode ='same')


# Load X and Y shader snippet arrays
filename = 'F:/Videogame_Assay/Snippets/x_shaders_snippets_around_touch_touching_light.csv'
x_snippets_open  = np.genfromtxt(filename, delimiter=',')
x_snippets = np.convolve(x_snippets_open[0,:], kernel)



filename = 'F:/Videogame_Assay/Snippets/y_shaders_snippets_around_touch_touching_light.csv'
y_snippets_open= np.genfromtxt(filename, delimiter=',')
y_snippets = np.convolve(y_snippets_open[0,:], kernel)

# Filter trials
all_trials = np.ones(num_trials, np.bool)
central_trials = (x_ball > 600) * (x_ball < 1100) * (y_ball > 450) * (y_ball < 850)

from_above_trials = (y_touch > y_ball)
from_below_trials = (y_touch < y_ball)
early_trials = trial_counts < 50
late_trials = trial_counts > 300
from_left_trials = (x_touch < x_ball)
from_right_trials = (x_touch > x_ball)

# Compute speed around touch (all trials)
filtered_trials = all_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed = np.sqrt(dx*dx + dy*dy)
avg_speed = np.nanmedian(speed, axis=0)
sem_avg_speed = stats.sem(speed, nan_policy='omit', axis=0)


# Compute speed around touch (from above trials)
filtered_trials = from_above_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
#avg_speed_from_above = np.nanmedian(np.sqrt(dx*dx + dy*dy), axis=0)
speed_above = np.sqrt(dx*dx + dy*dy)
avg_speed_from_above = np.nanmedian(speed_above, axis=0)
sem_above = stats.sem(speed_above, nan_policy='omit', axis=0)


filtered_trials = from_below_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
speed_below = np.sqrt(dx*dx + dy*dy)
avg_speed_from_below = np.nanmedian(speed_below, axis=0)
sem_below = stats.sem(speed_below, nan_policy='omit', axis=0)


plot_name = 'touching_light.png'

# Plot average speeds

figure_name = 'median_speed_above_below_' +plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(avg_speed), 'k')



plt.plot(avg_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),avg_speed-sem_avg_speed,avg_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')

plt.plot(avg_speed_from_above,color= 'r')
plt.fill_between(range(len(avg_speed_from_above)),avg_speed_from_above-sem_above,avg_speed_from_above+sem_above, alpha = 0.4, facecolor ='r')

plt.plot(avg_speed_from_below,color= 'b')
plt.fill_between(range(len(avg_speed_from_below)),avg_speed_from_below-sem_below,avg_speed_from_below+sem_below, alpha = 0.4, facecolor ='b')

ax.axes.get_yaxis().set_visible(True) 
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
#plt.plot(avg_speed, 'k')
#plt.plot(avg_speed_from_above, 'r')
#plt.plot(avg_speed_from_below, 'b')
plt.title("Avg Speed : Above (r) vs Below (b)")
#plt.show()

f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)




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
avg_speed_early = np.nanmedian(early_speed, axis=0)
sem_early = stats.sem(early_speed, nan_policy='omit', axis=0)


# Compute speed around touch (late trials)
filtered_trials = late_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
late_speed = np.sqrt(dx*dx + dy*dy)
avg_speed_late = np.nanmedian(late_speed, axis=0)
sem_late = stats.sem(late_speed, nan_policy='omit', axis=0)


# Plot average speeds

figure_name = 'median_speed_early_late_' +plot_name
  
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


plt.vlines(359, 0, max(avg_speed), 'k')
#plt.vlines(359, 0.0, 3.0, 'k')

plt.plot(avg_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),avg_speed-sem_avg_speed,avg_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(avg_speed_early,color= 'r')
plt.fill_between(range(len(avg_speed_early)),avg_speed_early-sem_early,avg_speed_early+sem_early, alpha = 0.4, facecolor ='r')

plt.plot(avg_speed_late,color= 'b')
plt.fill_between(range(len(avg_speed_late)),avg_speed_late-sem_late,avg_speed_late+sem_late, alpha = 0.4, facecolor ='b')


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
plt.title("Avg Speed : Early (r) vs Late (b)")
plt.show()
f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)




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
avg_speed_left = np.nanmedian(left_speed, axis=0)
sem_left = stats.sem(left_speed, nan_policy='omit', axis=0)

len(dx)
# Compute speed around touch (right trials)
filtered_trials = from_right_trials
dx = np.diff(x_snippets[filtered_trials], axis=1)
dy = np.diff(y_snippets[filtered_trials], axis=1)
right_speed = np.sqrt(dx*dx + dy*dy)
avg_speed_right = np.nanmedian(right_speed, axis=0)
sem_right = stats.sem(right_speed, nan_policy='omit', axis=0)
len(dx)

# Plot average speeds


figure_name = 'median_speed_left_right_' +plot_name
    
f,ax = plt.subplots(figsize=(9,5))

sns.set()
sns.set_style('white')
sns.axes_style('white')
sns.despine(left=False)


#plt.vlines(359, 0.001, 0.004, 'k')
plt.vlines(359, 0.0, max(avg_speed), 'k')

plt.plot(avg_speed,color= 'k')
plt.fill_between(range(len(sem_avg_speed)),avg_speed-sem_avg_speed,avg_speed+sem_avg_speed, alpha = 0.4, facecolor ='k')


plt.plot(avg_speed_left,color= 'r')
plt.fill_between(range(len(avg_speed_left)),avg_speed_left-sem_left,avg_speed_left+sem_left, alpha = 0.4, facecolor ='r')

plt.plot(avg_speed_right,color= 'b')
plt.fill_between(range(len(avg_speed_right)),avg_speed_right-sem_right,avg_speed_right+sem_right, alpha = 0.4, facecolor ='b')


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
plt.title("Avg Speed : left (r) vs right (b)")
plt.show()


f.tight_layout()

f.savefig(results_dir + figure_name, transparent=True)



