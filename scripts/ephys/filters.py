#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:44:34 2017

@author: atabak
"""

from scipy.signal import butter, filtfilt#, lfilter
import scipy.signal as signal

def butter_bandpass(lowcut, highcut, fs=30000, order=3, btype='bandstop'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a
  
def butter_lowpass(lowcut, fs=30000, order=3, btype='lowpass'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype=btype)
    return b, a


def butter_filter(data, lowcut, highcut, fs=30000, order=3, btype='bandstop'):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = filtfilt(b, a, data)
    return y

def butter_filter_lowpass(data, lowcut,  fs=30000, order=3, btype='lowpass'):
    b, a = butter_lowpass(lowcut, fs, order=order, btype=btype)
    y = filtfilt(b, a, data)
    return y

#Filter for extracellular recording
def highpass(data,BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500):
    b, a = signal.butter(BUTTER_ORDER,(passFreq/(sampleFreq/2), F_HIGH/(sampleFreq/2)),'pass')
    return signal.filtfilt(b,a,data)

def std_highpass(zero_mean):
    h_std=[]
    count=0
    for i in arange(128):
        high=highpass(zero_mean[i,:],BUTTER_ORDER=3, F_HIGH=14250,sampleFreq=30000.0,passFreq=500)
        high_std=np.std(high)
        high=None
        h_std.append(high_std)
    count += 1
    return np.array(h_std)
    
def std_lowpass(zero_mean,lowcut=250):
    l_std=[]
    count=0
    for i in arange(128):
        low=butter_filter_lowpass(zero_mean[i,:], lowcut,  fs=30000, order=3, btype='lowpass')
        low_std=np.std(low)
        low=None
        l_std.append(low_std)
    count += 1
    return np.array(l_std)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 400.0
    highcut = 1200.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype='bandstop')
        w, h = freqz(b, a, worN=5000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_filter(x, lowcut, highcut, fs, order=6, btype='bandpass')
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()