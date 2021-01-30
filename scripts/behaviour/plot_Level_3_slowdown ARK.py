import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy import stats
import statsmodels.api as sm
from scipy.signal import savgol_filter


# Load snippets
base_path = '/home/kampff/Downloads/respeedplotsupdates'
x_path = base_path + '/x_shaders_snippets_around_touch_all_rats_moving_light_pre.csv'
y_path = base_path + '/y_shaders_snippets_around_touch_all_rats_moving_light_pre.csv'
x_snippets  = np.genfromtxt(x_path, delimiter=',')
y_snippets  = np.genfromtxt(y_path, delimiter=',')

# Measure speed as usual
dx = np.diff(x_snippets, axis=1)
dy = np.diff(y_snippets, axis=1)
speeds = np.sqrt(dx*dx + dy*dy)   

# Measure noise and set threshold (however, this could be hard coded, 0.1 or so)
noise = np.std(speeds)
threshold = noise*3
#threshold = 0.1

# Find over threshold values
over_indices = speeds > threshold

# Set these to NaN
speeds[over_indices] = np.nan

# Measure approach and contact speeds
num_trials = speeds.shape[0]
trial_length = speeds.shape[1]
approach_speeds = np.zeros(num_trials)
contact_speeds = np.zeros(num_trials)
for i in range(num_trials):
    speed = speeds[i,:]
    approach_speeds[i] = np.nanmedian(speed[327:333])
    contact_speeds[i] = np.nanmedian(speed[354:360])

# Figure
diff_speeds = approach_speeds-contact_speeds
plt.plot(approach_speeds, contact_speeds, 'b.')
plt.plot(approach_speeds, approach_speeds, 'r.')
#plt.plot(contact_speeds, 'r.')
plt.show()

# Stats
np.mean(approach_speeds)
np.mean(contact_speeds)
stats.ttest_rel(approach_speeds, contact_speeds)
