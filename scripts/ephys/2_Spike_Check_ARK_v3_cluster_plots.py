# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:53:01 2018

@author: Kampff Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from filters import highpass

### Load and pre-process data

# Probe from superficial to deep electrode, left side is shank 11 (far back)
probe_map=np.array([[103,78,81,118,94,74,62,24,49,46,7],
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

# Load Data as uint16 from binary file, use memory mapping (i.e. do not load into RAM)
#   - use read-only mode "r+" to prevent overwriting the original file
filename = r'E:\Python_codes_11_06_19\test_session_33.2\2018_04_29-15_43\Amplifier.bin'
num_channels = 128
data = np.memmap(filename, dtype = np.uint16, mode = 'r')
num_samples = int(int(len(data))/num_channels)
freq = 30000
recording_time_sec = num_samples/freq
recording_time_min = recording_time_sec/60
reshaped_data = np.reshape(data,(num_samples,128))
#to have 128 rows
reshaped_data_T= reshaped_data.T
data = None
reshaped_data = None

# Extract data chunk for single channel
channel = 31
minutes = 60
seconds = minutes*60
num_samples_per_chunk = seconds*freq
channel_data = reshaped_data_T[channel,:num_samples_per_chunk]
reshaped_data_T = None

# Convert from interger values to microvolts, sub 32768 to go back to signed, 0.195 from analog to digital converter
channel_data_uV = (channel_data.astype(np.float32) - 32768) * 0.195
channel_data = None

# FILTERS (one ch at the time)
channel_data_highpass = highpass(channel_data_uV,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
#data_lowpass = butter_filter_lowpass(data_zero_mean[channel_number,:], lowcut=250,  fs=30000, order=3, btype='lowpass')
#channel_data_highpass = butter_filter(channel_data_uV, 500, 5000, fs=30000, order=3, btype='bandpass')

#lowcut = 500
#highcut = 2000
#channel_data_bandpass =  butter_filter(channel_data_uV, lowcut, highcut, fs=30000, order=3, btype='bandstop')

plt.figure()
#plt.plot(channel_data_bandpass[2000000:3100000])
plt.plot(channel_data_uV[0:300000])
#plt.plot(data_zero_mean[55][100000:105000])
plt.title('channel_' + str(channel))






# NEW CODE

# Determine high and low threshold
abs_channel_data_highpass = np.abs(channel_data_highpass)
sigma_n = np.median(abs_channel_data_highpass) / 0.6745
#sigma_n = np.std(abs_channel_data_highpass)
spike_threshold_hard = -5.0 * sigma_n
spike_threshold_soft = -3.0 * sigma_n

# Find spikes (peaks between high and low threshold crossings)
spike_start_times = []
spike_stop_times = []
spiking = False



# Are spikes downward or upward?
def threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft):
    
    spike_start_times = []
    spike_stop_times = []
    spiking = False
    
    for i, voltage in enumerate(channel_data_highpass):
        # Look for a new spike
        if(not spiking):
            if(voltage < spike_threshold_hard):
                spiking = True
                spike_start_times.append(i)
        # Track ongoing spike            
        else:
            # Keep track of max (negative) voltage until npo longer spiking
            if(voltage > spike_threshold_soft):
                spiking = False       
                spike_stop_times.append(i)
                  
    return spike_start_times, spike_stop_times

# Find threshold crossings
spike_start_times, spike_stop_times = threshold_crossing(channel_data_highpass,spike_threshold_hard,spike_threshold_soft)

# Find peak voltages and times
spike_peak_voltages = []
spike_peak_times = []
for start, stop in zip(spike_start_times,spike_stop_times):
    peak_voltage = np.min(channel_data_highpass[start:stop]) 
    peak_voltage_idx = np.argmin(channel_data_highpass[start:stop])
    spike_peak_voltages.append(peak_voltage)
    spike_peak_times.append(start + peak_voltage_idx)
  
# Remove too early and too late spikes
spike_starts = np.array(spike_start_times)
spike_stops = np.array(spike_stop_times)
peak_times = np.array(spike_peak_times)
peak_voltages = np.array(spike_peak_voltages)
good_spikes = (spike_starts > 100) * (spike_starts < (len(channel_data_highpass)-200))

# Select only good spikes
spike_starts = spike_starts[good_spikes]
spike_stops = spike_stops[good_spikes]
peak_times = peak_times[good_spikes]
peak_voltages = peak_voltages[good_spikes]

# Measure spike half widths
half_starts=[]
half_stops=[]
half_widths=[]
half_peak_voltages = np.array(peak_voltages)/2

# Find interpolated half_widths for all spikes
for i, peak_time in enumerate(peak_times):
    # Get half peak voltage level
    half_peak_voltage=half_peak_voltages[i]

    # Intialize loop variables
    crossed=False
    previous_voltage = 0
    
    # Start looping through all samples in data
    for i, voltage in enumerate(channel_data_highpass[(peak_time-30):(peak_time+30)]):
        # Set current sample (for convenience)
        current_sample = peak_time+i-30
        
        # Check for new crossing
        if(not crossed):
            if (voltage < half_peak_voltage):
                crossed = True
                
                sample_after_crossing = current_sample
                sample_before_crossing = current_sample - 1
                
                voltage_after_crossing = voltage
                voltage_before_crossing = previous_voltage
                
                # Measure interpolated "sample time" between before-after crossing samples
                voltage_magnitude_across_threshold = np.abs(voltage_after_crossing - voltage_before_crossing)
                voltage_magnitude_after_threshold = np.abs(voltage_after_crossing - half_peak_voltage)
                voltage_ratio = voltage_magnitude_after_threshold/voltage_magnitude_across_threshold            
                half_start = current_sample - voltage_ratio
        else:
            if (voltage > half_peak_voltage):
                crossed=False
                
                sample_after_crossing = current_sample
                sample_before_crossing = current_sample - 1
                
                voltage_after_crossing = voltage
                voltage_before_crossing = previous_voltage
                
                # Measure interpolated "sample time" between before-after crossing samples
                voltage_magnitude_across_threshold = np.abs(voltage_after_crossing - voltage_before_crossing)
                voltage_magnitude_after_threshold = np.abs(voltage_after_crossing - half_peak_voltage)
                voltage_ratio = voltage_magnitude_after_threshold/voltage_magnitude_across_threshold            
                half_stop = current_sample - voltage_ratio
                break
                
        # Store previous voltage
        previous_voltage = voltage

    # Store lists
    half_starts.append(half_start)
    half_stops.append(half_stop)

    # Compute half width
    half_widths.append(half_stop - half_start)

peak_half_widths = np.array(half_widths)

plt.figure(2)
cluster_A = peak_voltages < -125.0
cluster_B = peak_voltages > -125.0
plt.plot(peak_half_widths, peak_voltages, '.', color=[0,0,0,0.1])
plt.plot(peak_half_widths[cluster_A], peak_voltages[cluster_A], '.', color=[1,0,0,0.2])
plt.title('channel_' + str(channel))
plt.xlabel("peak_half_width")
plt.ylabel("peak_voltage")
#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
#plt.tick_params( axis='x',which='both',bottom=False,top=False,labelbottom=False)
#plt.box(on=None)




#        plt.axvline(x = vertical_line, color='k', linestyle='-', LineWidth=1)
#    
#        sns.set_style("white")
#        plt.xlabel("time")
#        plt.ylabel("Channel (μV)")
#        plt.title("Avg_highpass_abs"+ folder_name)
#        #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
#        plt.tick_params( axis='x',which='both',bottom=False,top=False,labelbottom=False)
#        plt.box(on=None)



# Plot all spikes
plt.figure(3)
spikes = np.zeros((len(peak_times), 300))
for i, s in enumerate(peak_times):
    spikes[i,:] = channel_data_highpass[(s-100):(s+200)]
plt.plot(spikes[range(0,len(peak_times), 2),:].T, '-', Color=[0,0,0,.002])

plt.ylim(-300, +200)
#plt.xlim(-100,200)
#plt.xticks(np.arange(-100, 200, step=50))
#sns.plt.xlim(0, None)
#avg_spike = np.mean(spikes, axis=0)
#plt.plot(avg_spike, '-', Color=[1,1,1,.5])
plt.title('channel_' + str(channel))

plt.figure(4)
spikes = np.zeros((len(peak_times),300))
for i, s in enumerate(peak_times):
    spikes[i,:] = channel_data_highpass[(s-100):(s+200)]
plt.figure()
plt.plot(spikes[range(0,len(peak_times), 2),:].T, '-', Color=[0,0,0,.002])

plt.ylim(-300, +200)
#sns.plt.xlim(0, None)
avg_spike = np.mean(spikes, axis=0)
plt.plot(avg_spike, '-', Color=[1,.5,1,.5])
plt.title('channel_' + str(channel))

# Only cluster A
plt.figure(5)
cluster_peak_times = peak_times[np.logical_not(cluster_A)]
spikes = np.zeros((len(cluster_peak_times),300))
for i, s in enumerate(cluster_peak_times):
    spikes[i,:] = channel_data_highpass[(s-100):(s+200)]
plt.figure()
plt.plot(spikes[range(0,len(cluster_peak_times), 2),:].T, '-', Color=[0,0,0,.002])

plt.ylim(-300, +200)
#sns.plt.xlim(0, None)
avg_spike = np.mean(spikes, axis=0)
plt.plot(avg_spike, '-', Color=[1,.5,1,.5])
plt.title('channel_' + str(channel))

# Only cluster B
plt.figure(6)
cluster_peak_times = peak_times[np.logical_not(cluster_B)]
spikes = np.zeros((len(cluster_peak_times),300))
for i, s in enumerate(cluster_peak_times):
    spikes[i,:] = channel_data_highpass[(s-100):(s+200)]
plt.figure()
plt.plot(spikes[range(0,len(cluster_peak_times), 2),:].T, '-', Color=[0,0,0,.002])

plt.ylim(-300, +200)
#sns.plt.xlim(0, None)
avg_spike = np.mean(spikes, axis=0)
plt.plot(avg_spike, '-', Color=[1,.5,1,.5])
plt.title('channel_' + str(channel))


#
#
#
#plt.figure(1)
#
#spike = channel_data_highpass[(spike_starts[10]-100):spike_starts[10]+200]
#plt.plot(spike,'k')
#plt.scatter(100,peak_voltages[0])
#
#
#spike1 = channel_data_highpass[(spike_starts[500]-100):spike_starts[500]+200]
#plt.plot(spike1,'k')
#plt.scatter(100,peak_voltages[500])
#
#spike2 = channel_data_highpass[(spike_starts[1500]-100):spike_starts[1500]+200]
#plt.plot(spike2,'k')
#plt.scatter(100,peak_voltages[1500])
#
#spike3 = channel_data_highpass[(spike_starts[2000]-100):spike_starts[2000]+200]
#plt.plot(spike3,'k')
#plt.scatter(100,peak_voltages[2000])
#
#
#
#fig, axs = plt.subplots(2, 2)
#axs[0, 0].plot(spike,'k')
#axs[0, 0].scatter(100,peak_voltages[10],color = 'red')
#axs[0,0].set_title('spike_10')
#
#
#axs[0, 1].plot(spike1,'k')
#axs[0, 1].scatter(100,peak_voltages[500],color = 'green')
#axs[0, 1].set_title('spike_500')
# 
#axs[1, 0].plot(spike2, 'k')
#axs[1, 0].scatter(100,peak_voltages[1500],color = 'blue')
#axs[1, 0].set_title('spike_1500')
#
#
#axs[1, 1].plot(spike3, 'k')
#axs[1, 1].scatter(100,peak_voltages[2000],color = 'magenta')
#axs[1, 1].set_title('spike_2000')
#
#
#






#np.random.seed(10)
#fig, ax = plt.subplots(figsize=(15, 5))
#
#for i in range(100):
#    spike = np.random.randint(0, wave_form.shape[0])
#    ax.plot(wave_form[spike, :])
#
#ax.set_xlim([0, 90])
#ax.set_xlabel('# sample', fontsize=20)
#ax.set_ylabel('amplitude [uV]', fontsize=20)
#ax.set_title('spike waveforms', fontsize=23)
#plt.show()


event_file='G:/Python_codes_11_06_19/test_session_33.2/2018_04_29-15_43/events/RatTouchBall.csv'
video_csv='G:/Python_codes_11_06_19/test_session_33.2/2018_04_29-15_43/Video.csv'
samples_for_frames_file_path='G:/Python_codes_11_06_19/test_session_33.2/2018_04_29-15_43/Analysis/samples_for_frames.csv'
touching_light=event_finder(event_file,video_csv,samples_for_frames_file_path)



spikes_around_event= []


touching_light= np.array(touching_light)


for touch in touching_light[:3]:
    touch_range = range(touch-30000,touch+30000)
    spike_list =[]
    for peak in peak_times:
        if peak in touch_range:
            spike_list.append(peak)    
    spikes_around_event.append(spike_list)


spikes_around_event= []


touching_light= np.array(touching_light)


for touch in touching_light[:3]:
    touch_range = range(touch-30000,touch+30000)
    spike_list =[]
    for peak in peak_times:
        if peak in touch_range:
            spike_list.append(1)  
        else:
            spike_list.append(0)
    spikes_around_event.append(spike_list)



#plt.figure()
#for trial in range(len(spikes_around_event)):
#    spikes_time=[i for i,x in enumerate(spikes_around_event[trial]) if x==1]
#    plt.vlines(spikes_time,trial,trial+1)
    



#
#touch= [30,50,80]
#
#fire =[ 10,14,23,31,36,39,45,48,51,68,11,98,76,81,3,105]
#
#shape=len(touch)
#
#
#empty= []
#
#
#for idx in touch:
#    event_range = range(idx-10,idx+10)
#    empty_list =[]
#    for each in fire:
#        if each in event_range:
#            empty_list.append(each)
#    empty.append(empty_list)






