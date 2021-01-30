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

x_path = touch_x_ephys
y_path = touch_y_ephys



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


speeds_smoothed= sliding_window(speeds, window_size=6)

# Measure median/mean
median_speed_cleaned = np.nanmedian(speeds, axis=0)
mean_speed_cleaned = np.nanmean(speeds, axis=0)

# Measure median/mean (smoothed)
median_speed_smoothed = np.nanmedian(speeds_smoothed, axis=0)
mean_speed_smoothed = np.nanmean(speeds_smoothed, axis=0)

# Plot
#plt.plot(speeds.T, 'r', alpha = 0.01)
plt.plot(mean_speed_cleaned, 'c')
plt.plot(mean_speed_smoothed, 'b')
plt.plot(median_speed_cleaned, 'm')
plt.plot(median_speed_smoothed, 'r')

plt.show()



tracking_crop = 'F:/Videogame_Assay/AK_33.2/2018_04_24-17_02/crop.csv'

tracking_shaders = 'F:/Videogame_Assay/AK_33.2/2018_04_24-17_02/events/Tracking.csv'


crop= np.genfromtxt(tracking_crop, delimiter=',')

shaders =np.genfromtxt(tracking_shaders, delimiter=None)