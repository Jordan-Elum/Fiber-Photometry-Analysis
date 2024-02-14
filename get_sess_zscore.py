# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:03:38 2023

@author: zweifeladmin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from get_downsampled_window import get_downsampled_window
import scipy.signal as ss

def get_sess_zscore(data, info, downsample_factor, save_path_traces, mov_win, trim_samples, sample_rate_raw):
    #downsample and trim 470 and 405 traces
    mov_win = mov_win
    trim_samples = trim_samples #cut off first ~n seconds to remove plug in artifact 
    data_trim = data[trim_samples:] #trim off first second of recording -- trimming entire block of rows (so no issue with TTL timestamps getting shifted)
    print('470 signal samples before and after trimming off first ' + str(round(trim_samples / sample_rate_raw)) +' seconds: ', data.GCaMP.shape[0], data_trim.GCaMP.values.shape[0])
    print('405 signal samples before and after trimming off first ' + str(round(trim_samples / sample_rate_raw)) +' seconds: ', data.ISOS.shape[0], data_trim.ISOS.values.shape[0])
    data_trim_gcamp = get_downsampled_window(downsample_factor, data_trim.GCaMP.values, mov_win)
    data_trim_isos = get_downsampled_window(downsample_factor, data_trim.ISOS.values, mov_win)
    session_time = np.linspace(data_trim['Time(sec)'].iloc[0], data_trim['Time(sec)'].iloc[-1], data_trim_gcamp.shape[0]) # to match downsample factor above
   
    #%% get raw 470 full trace with nan rows 
    plot_win = (0, data.shape[0]) #plot whole recording
    fig,axs = plt.subplots(3,1, figsize=(7,7), dpi = 1200)
    axs[0].plot(data['Time(sec)'][plot_win[0]:plot_win[1]], data.GCaMP[plot_win[0]:plot_win[1]], label = 'raw 470 untrimmed')
    axs[0].legend()
    
    plot_win = (0, data_trim.shape[0]) #plot whole recording
    axs[1].plot(data_trim['Time(sec)'][plot_win[0]:plot_win[1]], data_trim.GCaMP[plot_win[0]:plot_win[1]], label = 'raw 470 trimmed ' + str(round(trim_samples / sample_rate_raw)) + ' sec')
    axs[1].legend()
    
    plot_win = (0, data_trim_gcamp.shape[0]) #plot whole recording
    axs[2].plot(session_time[plot_win[0]:plot_win[1]], data_trim_gcamp[plot_win[0]:plot_win[1]], label = '470 smoothed and downsampled')
    axs[2].legend()
    
    plt.xlabel('Time (s)')
    plt.ylabel('mV')
    plt.title('Whole recording session' + '\n' + str(info.task.values) + str(info.mouseID.values) + str(info.block.values))
    plt.tight_layout()
    plt.savefig(save_path_traces + '\\' + str(info.task[0]) + '_' + str(info.mouseID[0]) + '_' + str(info.block[0])+'_raw_470_smooth_downsample_full_.png', bbox_inches='tight',dpi=1200)

    plot_win = (0, round(sample_rate_raw * 60)) #plot first n seconds
    fig,axs = plt.subplots(3,1, figsize=(7,7), dpi = 1200)

    axs[0].plot(data['Time(sec)'][plot_win[0]:plot_win[1]], data.GCaMP[plot_win[0]:plot_win[1]], label = 'raw 470 untrimmed')
    axs[0].legend()
    
    axs[1].plot(data_trim['Time(sec)'][plot_win[0]:plot_win[1]], data_trim.GCaMP[plot_win[0]:plot_win[1]], label = 'raw 470 trimmed ' + str(round(trim_samples / sample_rate_raw)) + ' sec')
    axs[1].legend()
    
    plot_win = (0, round(sample_rate_raw / downsample_factor) * 60) #plot first n seconds
    axs[2].plot(session_time[plot_win[0]:plot_win[1]], data_trim_gcamp[plot_win[0]:plot_win[1]], label = '470 smoothed and downsampled')
    axs[2].legend()
    
    plt.xlabel('Time (s)')
    plt.ylabel('mV')
    plt.title('First 60 s' + '\n' + str(info.task.values) + str(info.mouseID.values) + str(info.block.values))
    plt.tight_layout()
    plt.savefig(save_path_traces + '\\' + str(info.task[0]) + '_' + str(info.mouseID[0]) + '_' + str(info.block[0])+'_raw_470_smooth_downsample_first_min_.png', bbox_inches='tight',dpi=1200)
   
    #%% using 405 channel for curve fit/subtraction with regression followed by guppy exponential fit parameters
    fit_reg = np.polyfit(data_trim_isos, data_trim_gcamp, 1)
    fit_line_reg = np.multiply(fit_reg[0], data_trim_isos) + fit_reg[1]    
    x = session_time
    y = (data_trim_gcamp - fit_line_reg) / fit_line_reg #try subtracting fitted 405 first then fit exponential to get dff    
    def curveFitFn(x,a,b,c):
        return a+(b*np.exp(-(1/c)*x))
    p0 = [5,50,60]
    try:
        popt, pcov = curve_fit(curveFitFn, x, y, p0, maxfev=1800)
        fit_line = curveFitFn(x,*popt)
    except Exception as e:
        print('exponential curve fit fail, reshaping x or y to smaller array size for:', '\n' , info.values)
        x = x[0:min(x.shape,y.shape)[0]]
        y = y[0:min(x.shape,y.shape)[0]]
        popt, pcov = curve_fit(curveFitFn, x, y, p0, maxfev=1800)
        fit_line = curveFitFn(x,*popt)
    sess_dff = (y - fit_line) #using exponetial fit to remove slow baseline drift not captured by 405 fit
    sess_mean_dff = np.mean(sess_dff)
    sess_std_dff = np.std(sess_dff)
    sess_zscore = (sess_dff - np.mean(sess_dff)) / np.std(sess_dff)
    sess_mean_zscore = np.mean(sess_zscore)
    sess_fit_line = fit_line_reg
    #save session time, dff, and zscore in a csv
    df = pd.DataFrame({"time" : session_time, "dff" : sess_dff, "zscore" : sess_zscore})
    df.to_csv(save_path_traces + '\\' + str(info.task[0]) + '_' + str(info.mouseID[0]) + '_' + str(info.block[0])+ "_sess_dff_zscore.csv", index=False)
   
    #%% plot raw trace/fit
    fig,axs = plt.subplots(5,1, figsize=(12,12), dpi= 1200)
    axs[0].plot(session_time, data_trim_gcamp[0:len(sess_dff)], label = '470 raw trace')
    axs[0].plot(session_time, data_trim_isos[0:len(sess_dff)], label = '405 raw trace')
    axs[0].legend()
    axs[0].margins(x=0)
    
    axs[1].plot(session_time, data_trim_gcamp[0:len(sess_dff)], label = '470 raw trace')
    axs[1].plot(session_time, fit_line_reg[0:len(sess_dff)], label = '405 fitted trace', linewidth = 0.8, alpha = 0.8)
    axs[1].legend()
    axs[1].margins(x=0)
    
    axs[2].plot(session_time, y, label = 'dff', alpha = 0.8)    
    axs[2].plot(session_time, fit_line, label = 'fit (exponential)', alpha = 0.8)
    axs[2].legend()
    axs[2].margins(x=0)
    
    axs[3].plot(session_time, sess_dff, label = 'dff corrected')
    axs[3].legend()
    axs[3].margins(x=0)
    
    axs[4].plot(session_time, sess_zscore, label = 'zscore')
    axs[4].legend()
    axs[4].margins(x=0)
    fig.suptitle(str(info.task.values) + str(info.mouseID.values) + str(info.block.values), fontsize=16)
    plt.tight_layout()
    plt.xlabel('Time (s)')
    plt.savefig(save_path_traces + '\\' + str(info.task[0]) + '_' + str(info.mouseID[0]) + '_' + str(info.block[0])+'fit_dff_zscore_whole_session_.png', bbox_inches='tight',dpi=1200)
    plt.show()
    
    return(session_time, sess_dff, sess_zscore, info)
    