# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:31 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

#box and whisker  plot

 #q = np.quantile([50,14,15,20,13,70,72],[0.25,0.5,0.75])
 #iqr = q[2]-q[0]
# q[0]-1.5*iqr
 #q[2]+1*iqr lines on top of bars, values allowec before outlier
#plt.boxplot([50,14,15,20,13,70,106],whis=1) 106 outside


all_lenght_te = [[len(t) for t in s] for s in x_centroid_te]

rat_quantile = 

for s in arange(len(all_lenght_te)):
    

 test=   all_lenght_te[-1]

test = [all_lenght_te[0], all_lenght_te[1], all_lenght_te[2],all_lenght_te[4], all_lenght_te[6],all_lenght_te[7],all_lenght_te[8]]
plt.boxplot(test)


all_lenght_st= [[len(t) for t in s] for s in x_centroid_st]

test2 = [all_lenght_st[0], all_lenght_st[1], all_lenght_st[2],all_lenght_st[4], all_lenght_st[6],all_lenght_st[7],all_lenght_st[8]]
plt.boxplot(test2, showfliers=False)





