# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:17:27 2022

@author: zweifeladmin
"""
import numpy as np
import scipy.signal as ss

#downsample and smooth raw photometry signals
def get_downsampled_window(downsample_factor,zarray,mov_win):
    #smooth raw FP trace: (linear digital filter) for smoothing
    signal = zarray
    filter_window = mov_win
    b = np.divide(np.ones((filter_window,)), filter_window)
    a = 1
    signal_smooth = ss.filtfilt(b, a, signal)

    # Downsample and average via a moving window mean
    arr = signal_smooth
    min1 = arr.shape[0]
    N = downsample_factor # Average every n samples into 1 value
    dwnsmpl_signal = []
    for i in range(0, min1, N):
        # print(i)
        dwnsmpl_signal.append(np.mean(arr[i:i+N-1])) # This is the moving window mean    
    print('number samples before and after smoothing and downsampling: ', len(signal), ' --> ', len(dwnsmpl_signal))
    
    return(np.array(dwnsmpl_signal))
