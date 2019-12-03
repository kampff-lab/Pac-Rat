# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:56:19 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import os
#os.sys.path.append('/home/kampff/Repos/Pac-Rat/libraries')
os.sys.path.append('D:/Repos/Pac-Rat/libraries')
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import behaviour_library as behaviour
import parser_library as prs
from matplotlib.colors import PowerNorm  
from matplotlib.colors import LogNorm 
from pylab import *
from matplotlib.ticker import LogFormatterExponent
import seaborn as sns 
import collections
from collections import Counter


rat_summary_table_path =r'F:/Videogame_Assay/AK_33.2_Pt.csv'
hardrive_path = r'F:/' 
RAT_ID = 'AK_33.2'


#Level_0 = prs.Level_0_paths(rat_summary_table_path)
#Level_1_6000_3000 = prs.Level_1_paths_6000_3000(rat_summary_table_path)
#Level_1_10000 = prs.Level_1_paths_10000(rat_summary_table_path)
#Level_1_20000 = prs.Level_1_paths_20000(rat_summary_table_path)
Level_2_pre = prs.Level_2_pre_paths(rat_summary_table_path)


#saving a Trial_idx_csv containing the idx of start-end-touch 0-1-2
sessions_subset = Level_2_pre


clip_00_path = r'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/Clips/Clip00.avi'
clip_00 = cv2.VideoCapture(clip_00_path)
num_frames = int(clip_00.get(cv2.CAP_PROP_FRAME_COUNT))

#add the tot frames of the video to the list of idx so that after the diff there is the tim of the last behaviour annotated

clip_path = r'F:/Videogame_Assay/AK_33.2/2018_04_06-15_13/Clips_annotation/Clip00.csv'
clip = np.genfromtxt (clip_path, delimiter = ',', dtype = str, usecols=0)
time = np.genfromtxt (clip_path, delimiter = ',', dtype = int, usecols=1)

etho_list = ['wlk','run','gal','scr','gro','toi','sle','sni','pos','stl','lin','rea','otu','mar','jum','poh','poc','por','pok','stb','fre','ale','sta','hop','bwl']


count_behaviours = Counter(clip)
test =  list(count_behaviours) 

list_behaviours = count_behaviours.items()  

test2= Counter(dict(list_behaviours)) 

 # dictionary sorted by key
test3= collections.OrderedDict(sorted(count_behaviours.items(), key=lambda t: t[0]))
OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])

>>> # dictionary sorted by value
test4 = collections.OrderedDict(sorted(clip.items(), key=lambda t: t[1]))
OrderedDict([('pear', 1), ('orange', 2), ('banana', 3), ('apple', 4)])

# Given a list of words, return a dictionary of
# word-frequency pairs.



def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))



test = wordListToFreqDict(clip.tolist())

# Sort a dictionary of word-frequency pairs in
# order of descending frequency.

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


# Sort a dictionary of word-frequency pairs in
# order of descending frequency.

aux =  sortFreqDict(test)


x,y = zip(*aux)
plt.bar(y,x)




