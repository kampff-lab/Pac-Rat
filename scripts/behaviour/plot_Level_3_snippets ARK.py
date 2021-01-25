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
speed = np.sqrt(dx*dx + dy*dy)   

# Measure noise and set threshold (however, this could be hard coded, 0.1 or so)
noise = np.std(speed)
threshold = noise*5
#threshold = 0.1

# Find over threshold values
over_indices = speed > noise

# Set these to NaN
speed[over_indices] = np.nan

# Measure median/mean
median_speed_cleaned = np.nanmedian(speed, axis=0)
mean_speed_cleaned = np.nanmean(speed, axis=0)

# Plot
#plt.plot(speed.T, 'r', alpha = 0.01)
plt.plot(median_speed_cleaned)
plt.show()
