# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
from scipy import stats 


import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
# Reload modules
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random 

alpha = [0.6,1,0.3,0.4, 0.7]
x_coordinate = [[1,2,3,4,5,6,7,8,9,10,11]]
y_coordiinate = [5,7,9,20,20]

x = np.arange(10)
y = np.arange(30)

colors = cm.rainbow(np.linspace(alpha))

for i in arange(len(alpha)):
    
    plt.scatter(x_coordinate_final,y_coordinate_final, c = random_test, cmap="bwr",s=100)
    plt.colorbar()
    plt.hlines(4808,0,12)


#x_coordinate_final = np.repeat(x_coordinate,11)




def plotting_probe_coordinates():
    
    probe_x_coordinates = np.array(list(np.arange(1,12))*11)

    #find probe y values 
    
    #l = [3198,3004,2989,2831,2874,3127,3483,3971,4108,4276]
    #first = [4808]
    #l_diff=np.subtract(l,first)


    shank_1 =np.arange(0,2640,240)
    shank_2 =np.arange(0,2310,210) +532
    shank_3 =np.arange(0,2200,200) +700
    shank_4 =np.arange(0,2200,200) +837
    shank_5 =np.arange(0,1870,170) +1325
    shank_6 = np.arange(0,1760,160)+1681
    shank_7 =np.arange(0,1540,140) +1934
    shank_8 =np.arange(0,1320,120) +1977
    shank_9 =np.arange(0,1540,140) +1819
    shank_10 =np.arange(0,1540,140)+1804
    shank_11=np.arange(0,1540,140) +1610


#y_coordinate_final = np.hstack((shank_11,shank_10,shank_9,shank_8,shank_7,shank_6,shank_5,shank_4,shank_3,shank_2,shank_1))

    
    y_by_flatten_probe = []

    
    for i in range(11):
        
        sort_shank_11= sorted(shank_11,reverse=True)
        y_by_flatten_probe.append(sort_shank_11[i])
        
        sort_shank_10= sorted(shank_10,reverse=True)
        y_by_flatten_probe.append(sort_shank_10[i])  
    
        sort_shank_9= sorted(shank_9,reverse=True)
        y_by_flatten_probe.append(sort_shank_9[i])
    
        sort_shank_8= sorted(shank_8,reverse=True)
        y_by_flatten_probe.append(sort_shank_8[i])
        
        sort_shank_7= sorted(shank_7,reverse=True)
        y_by_flatten_probe.append(sort_shank_7[i])    
        
        sort_shank_6= sorted(shank_6,reverse=True)
        y_by_flatten_probe.append(sort_shank_6[i])   
        
        sort_shank_5= sorted(shank_5,reverse=True)
        y_by_flatten_probe.append(sort_shank_5[i])    
           
        sort_shank_4= sorted(shank_4,reverse=True)
        y_by_flatten_probe.append(sort_shank_4[i])
        
        sort_shank_3= sorted(shank_3,reverse=True)
        y_by_flatten_probe.append(sort_shank_3[i])
        
        sort_shank_2= sorted(shank_2,reverse=True)
        y_by_flatten_probe.append(sort_shank_2[i])
        
        sort_shank_1= sorted(shank_1,reverse=True)
        y_by_flatten_probe.append(sort_shank_1[i])
        

    return probe_x_coordinates, y_by_flatten_probe


x,y = plotting_probe_coordinates()
    
#test = shank_11.T
#
#random_test = np.array(np.random.randint(100, size=(1, 121)).tolist()).flatten()
#
#
#
#l = [3198,3004,2989,2831,2874,3127,3483,3971,4108,4276]
#first = [4808]
#l_diff=np.subtract(l,first)
