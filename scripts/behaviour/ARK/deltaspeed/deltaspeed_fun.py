import numpy as np
import matplotlib.pyplot as plt

# Load trial table
filename = 'F:/Videogame_Assay/Trial_table_final_level_2_touching_light.csv'
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

x_post = trial_table[:,10]
y_post = trial_table[:,11]

x_ball = trial_table[:,16]
y_ball = trial_table[:,17]

speed_pre = np.sqrt((x_pre - x_touch)**2 + (y_pre - y_touch)**2)
speed_post = np.sqrt((x_post - x_touch)**2 + (y_post - y_touch)**2)

# Compute delta- and ratio- speeds for each trial
deltaspeed = speed_post - speed_pre
ratiospeed = speed_post/speed_pre

# Compute deltangles for each trial
heading_pre = np.arctan2((y_touch - y_pre), (x_touch - x_pre)) * 180 / np.pi
heading_post = np.arctan2((y_post - y_touch), (x_post - x_touch)) * 180 / np.pi
deltangle = heading_post - heading_pre

# Setup filters
central_trials = (x_ball > 600) * (x_ball < 1100) * (y_ball > 450) * (y_ball < 850)
from_above_trials = (y_ball - y_pre) > 100
from_below_trials = (y_ball - y_pre) < -100
moving_trials = speed_pre > 100
accel_trials = ratiospeed > 1.0

# Plot delta speed
plt.plot(trial_counts, deltaspeed, 'b.')
plt.show()

# Plot deltangles
plt.plot(trial_counts, deltangle, 'g.')
plt.show()

# Plot ratio speed
plt.plot(trial_counts, ratiospeed, 'b.')
plt.ylim(0, 3)
plt.show()


# Specify valid trial counters
valid_trial_counts_above = trial_counts[from_above_trials * central_trials * moving_trials]
valid_trial_counts_below = trial_counts[from_below_trials * central_trials * moving_trials]

# Plot delta speed
deltaspeed_above = deltaspeed[from_above_trials * central_trials * moving_trials]
deltaspeed_below = deltaspeed[from_below_trials * central_trials * moving_trials]
plt.plot(valid_trial_counts_above, deltaspeed_above, 'r.')
plt.plot(valid_trial_counts_below, deltaspeed_below, 'b.')
plt.show()
print(np.mean(deltaspeed_above))
print(np.mean(deltaspeed_below))

# Plot delta angle
deltangle_above = deltangle[from_above_trials * central_trials * moving_trials]
deltangle_below = deltangle[from_below_trials * central_trials * moving_trials]
plt.plot(valid_trial_counts_above, deltangle_above, 'r.')
plt.plot(valid_trial_counts_below, deltangle_below, 'g.')
plt.show()
print(np.mean(deltangle_above))
print(np.mean(deltangle_below))


# Plot ratio speed
ratiospeed_above = ratiospeed[from_above_trials * central_trials * moving_trials]
ratiospeed_below = ratiospeed[from_below_trials * central_trials * moving_trials]
plt.plot(valid_trial_counts_above, ratiospeed_above, 'r.')
plt.plot(valid_trial_counts_below, ratiospeed_below, 'b.')
plt.show()
print(np.mean(ratiospeed_above))
print(np.mean(ratiospeed_below))

#prespeed_above = speed_pre[from_above_trials * central_trials]
#prespeed_below = speed_pre[from_below_trials * central_trials]
#plt.plot(prespeed_above, 'r.')
#plt.plot(prespeed_below, 'b.')
#plt.show()
#print(np.mean(prespeed_above))
#print(np.mean(prespeed_below))
#
#postspeed_above = speed_post[from_above_trials * central_trials]
#postspeed_below = speed_post[from_below_trials * central_trials]
#plt.plot(postspeed_above, 'r.')
#plt.plot(postspeed_below, 'b.')
#plt.show()
#print(np.mean(postspeed_above))
#print(np.mean(postspeed_below))
#

accel_trials = (deltaspeed < 50.0) * ((deltaspeed > -50.0))
plt.plot(x_ball[accel_trials], y_ball[accel_trials], 'r.')
plt.plot(x_ball[accel_trials == False], y_ball[accel_trials == False], 'b.')
plt.show()

accel_trials = (deltaspeed < 50.0) * ((deltaspeed > -50.0))
plt.plot(speed_pre[accel_trials], speed_post[accel_trials], 'r.')
plt.show()

#