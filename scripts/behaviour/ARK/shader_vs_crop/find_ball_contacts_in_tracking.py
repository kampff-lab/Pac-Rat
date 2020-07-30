import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse

# Load timestamp and position strings
filename = '/home/kampff/Dropbox/LC_THESIS/ARK/shader_vs_crop/BallTracking.csv'
split_data = np.genfromtxt(filename, delimiter=[33,100], dtype='unicode')
timestamp_strings = split_data[:,0]
positions_strings = split_data[:,1]

# Find positions
for index, s in enumerate(positions_strings):
    tmp = s.replace('(', '')
    tmp = tmp.replace(')', '')
    tmp = tmp.replace('\n', '')
    tmp = tmp.replace(' ', '')
    positions_strings[index] = tmp
positions = np.genfromtxt(positions_strings, delimiter=',', dtype=float)
dx = np.diff(positions[:,0], prepend=[0])
dy = np.diff(positions[:,1], prepend=[0])
speed = np.sqrt(dx*dx + dy*dy)
moving = np.int32(speed > 0)
starts = np.where(np.diff(moving) == 1)[0]

# Convert to elapsed seconds
elapsed = []
start_time = parse(timestamp_strings[0])
for ts in timestamp_strings:
    current_time = parse(ts)
    delta_time = current_time - start_time
    elapsed.append(delta_time.total_seconds())
elapsed = np.array(elapsed)

# Find timestamp jumps (i.e. start indices)
deltas = np.diff(elapsed, prepend=[0])
deltas[0] = 11 # Added so the first index is considered a "start"
#plt.plot(deltas, '.')
#plt.show()
start_indices = np.where(deltas > 1)[0]
start_timestamp_strings = timestamp_strings[start_indices]

# Find contact indices
dx = np.diff(positions[:,0], prepend=[0])
dy = np.diff(positions[:,1], prepend=[0])
contact_indices = []
for i in start_indices:
    dx_trial = dx[(i+1):]
    dy_trial = dy[(i+1):]
    trial_end = len(dx_trial)
    j = 0
    while j <  trial_end:
        if((dx_trial[j] > 0) or (dy_trial[j] > 0)):
            contact_indices.append(i + j)
            j = trial_end
        else:
            j = j + 1
contact_indices = np.array(contact_indices)
contact_timestamp_strings = timestamp_strings[contact_indices]

#FIN