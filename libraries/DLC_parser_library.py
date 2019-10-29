# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:34:09 2019

@author: KAMPFF-LAB-ANALYSIS3
"""

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

from scipy import interpolate
from scipy.interpolate import interp1d

#DLC tracking file

#after removing first row
#row 1,0 = bodyparts 
#'Nose' = 0,1 = x
# 0,2  = y
# 2,3 likelihood nose

#
#
#dlc_vieo_path = 'D:/cuttlefish/Cuttlefish_butts_DLC/10-22T15_37_43_CROPPED.avi'
#dlc_tracking_path = 'D:/cuttlefish/Cuttlefish_butts_DLC/BEST_croppedDeepCut_resnet50_Cuttle-ShuttleOct14shuffle1_250000.csv'
#
#dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)
#
#
#x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
#y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])
#
#
#
#cleaned_x = [x for x in x_nan if str(x) != 'nan']
#cleaned_y = [x for x in y_nan if str(x) != 'nan']
#
##cuttlefish
#hist2d(cleaned_x, cleaned_y, bins=250, norm = LogNorm(), cmap='viridis',vmin=10e0, vmax=10e3)



#rat


#
#dlc_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_07-15_42/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv'
#
#dlc_tracking= np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)
#
#
#x_nan = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
#y_nan =  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])
#
#
#
#cleaned_x = [x for x in x_nan if str(x) != 'nan']
#cleaned_y = [x for x in y_nan if str(x) != 'nan']

'F:/Videogame_Assay/AK_33.2/2018_04_08-10_55/events/Trial_idx.csv'
'F:/Videogame_Assay/AK_33.2/2018_04_08-10_55/events/Ball_coordinates.csv'

#nose dlc  x = 1  / y = 2 / likelihood = 3
#tail base dlc   x = 10  / y = 11  / likelihood = 12
def DLC_coordinates_correction(session, crop_size = 640, dlc_x_column = 1, dlc_y_column = 2, dlc_likelihood_column = 3):
    
    # Centroid path
    centroid_tracking_path = os.path.join(hardrive_path + session + '/crop.csv')
    # Load Centroid tracking
    centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)
    
    
    # Select x and y from centroid file 
    centroid_x = centroid_tracking[:, 0] 
    centroid_y = centroid_tracking[:, 1] 
    
    # Load DLC tracking
    dlc_tracking_path = os.path.join(hardrive_path + session + '/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv')
    dlc_tracking = np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)
    
    # Select x and y from DLC file
    dlc_x = dlc_tracking[:, dlc_x_column] 
    dlc_y = dlc_tracking[:, dlc_y_column] 
    #dlc_x_tail_base = dlc_tracking[:, 10] 
    #dlc_y_tail_base = dlc_tracking[:, 11]
    
    # Center DLC coordinates in crop window
    dlc_centered_x = dlc_x - (crop_size / 2)
    dlc_centered_y = dlc_y - (crop_size / 2) 
    #dlc_centered_x_tail_base = dlc_x_tail_base - (crop_size / 2)
    #dlc_centered_y_tail_base = dlc_y_tail_base - (crop_size / 2) 
    
    # Correct coordinates base on crop region centroid
    dlc_correct_x = centroid_x + dlc_centered_x
    dlc_correct_y = centroid_y + dlc_centered_y
    #dlc_correct_x_tail_base = centroid_x + dlc_centered_x_tail_base
    #dlc_correct_y_tail_base = centroid_y + dlc_centered_y_tail_base
    
    # Fill with Nan x and y if the likehood of dlc is less than 0.9
    x_nan = np.where(dlc_tracking[:,dlc_likelihood_column]<= 0.9999, np.NaN, dlc_correct_x)
    y_nan =  np.where(dlc_tracking[:,dlc_likelihood_column]<= 0.9999, np.NaN, dlc_correct_y)
    #x_nan_tail_base = np.where(dlc_tracking[:,12]<= 0.9999, np.NaN, dlc_correct_x_tail_base)
    #y_nan_tail_base =  np.where(dlc_tracking[:,12]<= 0.9999, np.NaN, dlc_correct_y_tail_base)
        
    return x_nan ,y_nan
        


    x_nan_tail_base = np.where(dlc_tracking[:,12]<= 0.9999, np.NaN, dlc_correct_x_tail_base)
    y_nan_tail_base =  np.where(dlc_tracking[:,12]<= 0.9999, np.NaN, dlc_correct_y_tail_base)




    





### START DLC IMPORT and CLEAN HERE! ###

crop_size = 640
centroid_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_08-10_55/crop.csv'
dlc_tracking_path = 'F:/Videogame_Assay/AK_33.2/2018_04_08-10_55/cropDeepCut_resnet50_Pac-RatSep13shuffle1_250000.csv'


# Load Centroid tracking
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)

# Select x and y from centroid file 
centroid_x = centroid_tracking[:, 0] 
centroid_y = centroid_tracking[:, 1] 

# Load DLC tracking
dlc_tracking = np.genfromtxt(dlc_tracking_path, delimiter = ',', skip_header = 3, dtype = float)

# Select x and y from DLC file
dlc_x_nose = dlc_tracking[:, 1] 
dlc_y_nose = dlc_tracking[:, 2] 
dlc_x_tail_base = dlc_tracking[:, 10] 
dlc_y_tail_base = dlc_tracking[:, 11]

# Center DLC coordinates in crop window
dlc_centered_x_nose = dlc_x_nose - (crop_size / 2)
dlc_centered_y_nose = dlc_y_nose - (crop_size / 2) 
dlc_centered_x_tail_base = dlc_x_tail_base - (crop_size / 2)
dlc_centered_y_tail_base = dlc_y_tail_base - (crop_size / 2) 

# Correct coordinates base on crop region centroid
dlc_correct_x_nose = centroid_x + dlc_centered_x_nose
dlc_correct_y_nose = centroid_y + dlc_centered_y_nose
dlc_correct_x_tail_base = centroid_x + dlc_centered_x_tail_base
dlc_correct_y_tail_base = centroid_y + dlc_centered_y_tail_base

# Fill with Nan x and y if the likehood of dlc is less than 0.9
x_nan_nose = np.where(dlc_tracking[:,3]<= 0.9999, np.NaN, dlc_correct_x_nose)
y_nan_nose =  np.where(dlc_tracking[:,3]<= 0.9999, np.NaN, dlc_correct_y_nose)
x_nan_tail_base = np.where(dlc_tracking[:,12]<= 0.9999, np.NaN, dlc_correct_x_tail_base)
y_nan_tail_base =  np.where(dlc_tracking[:,12]<= 0.9999, np.NaN, dlc_correct_y_tail_base)


##






# Compute speed
dx = np.diff(x_nan_nose, prepend=[0])
dy = np.diff(y_nan_nose, prepend=[0])
speed = np.sqrt(dx*dx + dy*dy)

# Compute threshold for "bad" speeds
speed_threshold = 3 * np.nanstd(speed)

# Remove bad speeds
x_filtered_nan_nose = np.where(speed > speed_threshold, np.NaN, x_nan_nose)
y_filtered_nan_nose = np.where(speed > speed_threshold, np.NaN, y_nan_nose)
x_filtered_nan_tail_base = np.where(speed > speed_threshold, np.NaN, x_nan_tail_base)
y_filtered_nan_tail_base = np.where(speed > speed_threshold, np.NaN, y_nan_tail_base)

# Interpolate across NaNs
x_nose = pad(x_filtered_nan_nose)
y_nose =  pad(y_filtered_nan_nose)
x_tail_base = pad(x_filtered_nan_tail_base)
y_tail_base =  pad(y_filtered_nan_tail_base)

# Plot interpolation
plt.figure()
plt.plot(x_nose, 'g')
low_likelihood = np.where(dlc_tracking[:,3] < 0.9999)[0]
plt.plot(low_likelihood,dlc_correct_x_nose[low_likelihood], 'r.', alpha=0.5)
high_likelihood = np.where(dlc_tracking[:,3] > 0.9999)[0]
plt.plot(high_likelihood,dlc_correct_x_nose[high_likelihood], 'b.', alpha=0.5)
plt.show()





# Filter



# Plot
plt.figure()
plt.plot(x_nan_nose, y_nan_nose, 'g')
plt.plot(x_filtered_nan_nose, y_filtered_nan_nose, 'b')
plt.plot(x_nose, y_nose, 'r')
plt.show()

# Compute speed
dx = np.diff(x_nose, prepend=[0])
dy = np.diff(y_nose, prepend=[0])
speed = np.sqrt(dx*dx + dy*dy)
plt.plot(speed)
plt.show()


# Plot DLC likelihood ranges
plt.figure()
low_likelihood = np.where(dlc_tracking[:,3] < 0.999)[0]
plt.plot(dlc_correct_x_nose[low_likelihood], dlc_correct_y_nose[low_likelihood], 'r.', alpha=0.1)
high_likelihood = np.where(dlc_tracking[:,3] > 0.999)[0]
plt.plot(dlc_correct_x_nose[high_likelihood], dlc_correct_y_nose[high_likelihood], 'b.', alpha=0.1)
plt.show()

plt.figure()
low_likelihood = np.where(dlc_tracking[:,3] < 0.9999)[0]
plt.plot(low_likelihood,dlc_correct_x_nose[low_likelihood], 'r.', alpha=0.1)
high_likelihood = np.where(dlc_tracking[:,3] > 0.9999)[0]
plt.plot(high_likelihood,dlc_correct_x_nose[high_likelihood], 'b.', alpha=0.1)
plt.show()

plt.figure()
low_likelihood = np.where(dlc_tracking[:,12] < 0.9)[0]
plt.plot(low_likelihood,dlc_correct_x_tail_base[low_likelihood], 'r.', alpha=0.1)
high_likelihood = np.where(dlc_tracking[:,12] > 0.9)[0]
plt.plot(high_likelihood,dlc_correct_x_tail_base[high_likelihood], 'b.', alpha=0.1)
plt.show()




# Plot DLC likelihood ranges
plt.figure()
plt.plot(x_nan_nose, y_nan_nose, '.')
plt.show()


# Plot
plt.figure()
plt.plot(x_correct_nose, y_correct_nose)
plt.show()

plt.plot(dlc_tracking[:,3], 'k.')
plt.show()

#remove nans
#cleaned_x = [x for x in x_nose if str(x) != 'nan']
#cleaned_y = [x for x in y_nose if str(x) != 'nan']





# interpolate 1



x_nan_int_2 = np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,1])
y_nan_int_2=  np.where(dlc_tracking[:,3]<= 0.99, np.NaN, dlc_tracking[:,2])





ok = ~np.isnan(x_correct)
xp = ok.ravel().nonzero()[0]
fp = x_correct[~np.isnan(x_correct)]
x  = np.isnan(x_correct).ravel().nonzero()[0]
x_correct[np.isnan(x_correct)] = np.interp(x, xp, fp)

ok2 = ~np.isnan(y_correct)
xp2 = ok2.ravel().nonzero()[0]
fp2 = y_correct[~np.isnan(y_correct)]
x2  = np.isnan(y_correct).ravel().nonzero()[0] 
y_correct[np.isnan(y_correct)] = np.interp(x2, xp2, fp2)










def pad(data):
    out = np.copy(data)
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = out[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    out[bad_indexes] = interpolated
    return out



#import numpy as np
#nan = np.nan
#
#A = np.array([1, nan, nan, 2, 2, nan, 0])
#
#ok = ~np.isnan(A)
#xp = ok.ravel().nonzero()[0]
#fp = A[~np.isnan(A)]
#x  = np.isnan(A).ravel().nonzero()[0]
#
#A[np.isnan(A)] = np.interp(x, xp, fp)
#
#print (A)
#[1.         1.33333333 1.66666667 2.         2.         1.
# 0.        ]


centroid_tracking_path = r'F:/Videogame_Assay/AK_33.2/2018_03_27-14_34/crop.csv'
centroid_tracking = np.genfromtxt(centroid_tracking_path, delimiter = ',', dtype = float)


centroid_tracking_wo_nan= centroid_tracking[~np.isnan(centroid_tracking).any(axis=1)]
x_centroid = centroid_tracking_wo_nan[:,0]
y_centroid = centroid_tracking_wo_nan[:,1]


from matplotlib.colors import PowerNorm
        
from matplotlib.colors import LogNorm        
        
plt.hist2d(x_centroid, y_centroid, bins=150, norm=LogNorm()) #norm=PowerNorm(0.3)






norm=PowerNorm(0.3)

x_centroid = centroid_tracking[:,0]
y_centroid = centroid_tracking[:,1]
plt.plot(x_centroid,y_centroid, marker ='o',color= [0,0,0,0.002])


nan_finder = np.argwhere(np.isnan(centroid_tracking))

nan_finder_x= np.argwhere(np.isnan(x_centroid))
nan_finder_y = np.argwhere(np.isnan(y_centroid))


tracking_diff=np.diff(centroid_tracking)


plt.hexbin(x_centroid,y_centroid)


plt.hist2d(x_centroid,y_centroid)

arr = np.array(data)

df = pd.DataFrame(centroid_tracking)

sns.heatmap(df, cmap='coolwarm')


