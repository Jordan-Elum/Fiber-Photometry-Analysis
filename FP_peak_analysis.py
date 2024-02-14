# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:36:37 2023

@author: Jordan Elum
"""

import numpy as np
from scipy import io
from numpy import load
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import chirp, find_peaks, peak_widths
import seaborn as sns
import glob

def FP_peak_analysis(zscore, samples_sec):
    x = zscore
    prom = 0.4
    dist = None
    thresh = 0.001
    pos_peaks_all = scipy.signal.find_peaks(x, height=None, threshold=thresh, distance=dist, 
                            prominence=prom, width=None, wlen=None, rel_height=None, plateau_size=None)
    pos_peaks=pos_peaks_all[0]
    results_half = peak_widths(x, pos_peaks_all[0], rel_height=0.5)
    results_half[0]  # widths
    results_full = peak_widths(x, pos_peaks_all[0], rel_height=1)
    results_full[0] # widths
    total_time = (zscore.shape[0] / (samples_sec)) / 60 #in minutes
    event_per_min = pos_peaks.shape[0] / total_time
    peak_amp = np.mean(x[pos_peaks])
    peak_width = np.mean(results_half[0]) / (samples_sec)
    print('events per min, peak_amp_mean: ', 'peak_width (s): ', event_per_min, peak_amp, peak_width)
    #%%
    plt.figure(dpi=1200)
    min_rng = 0
    max_rng = pos_peaks.shape[0] - 1 
    plt.plot(np.arange(pos_peaks[min_rng], pos_peaks[max_rng]), x[pos_peaks[min_rng]:pos_peaks[max_rng]], linewidth=0.8, color='black')
    plt.plot(pos_peaks[min_rng:max_rng], x[pos_peaks][min_rng:max_rng], "o", markersize = 2,  color = 'red')
    plt.hlines(*results_half[1:], color='gray', alpha=0.6, label='width', linewidth=2)
    plt.margins(x=0)
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None)
    plt.savefig(peak_save_path + '\\' +  '_peaks_whole_trace' +  '_' + '.svg',  transparent=True, bbox_inches='tight')
    plt.savefig(peak_save_path + '\\' +  '_peaks_whole_trace' +  '_' + '.png', bbox_inches='tight')
    plt.show()
    
    plt.figure(dpi=1200)
    min_rng = 0
    max_rng = pos_peaks.shape[0] - 1 
    plt.plot(np.arange(pos_peaks[min_rng], pos_peaks[max_rng]), x[pos_peaks[min_rng]:pos_peaks[max_rng]], linewidth=0.8, color='black')
    plt.plot(pos_peaks[min_rng:max_rng], x[pos_peaks][min_rng:max_rng], "o", markersize = 2,  color = 'red')
    plt.hlines(*results_half[1:], color='gray', alpha=0.6, label='width', linewidth=2)
    plt.margins(x=0)
    plt.xlim(0 , 800)
    sns.despine(top=True, right=True, left=False, bottom=False, offset=None)
    plt.savefig(peak_save_path + '\\' +  '_peaks_representative_width' +  '_' + '.svg',  transparent=True, bbox_inches='tight')
    plt.savefig(peak_save_path + '\\' +  '_peaks_representative_width' +  '_' + '.png', bbox_inches='tight')
    plt.show()    
    return(event_per_min, peak_amp, prom, dist, thresh, peak_width)

#%% Read in processed z-scored FP recording files
zscores_path = input('Enter the absolute path of the directory in which the FP recording z-score files are stored (no quotes): ')
zscore_files = []
zscore_files = glob.glob(zscores_path + "/**/*_sess_dff_zscore.csv", recursive = True)  
print('\n', 'Processed z-scores filenames list:\n', *zscore_files, sep = '\n')

peak_save_path = input('Enter the absolute path of the directory in which the peaks output data should be saved (no quotes): ')
samples_sec = 10.133333333333333
peak_df = pd.DataFrame()
for data_file in zscore_files:
    sess_data = pd.read_csv(data_file)
    #call FP_peak_analysis function to process FP recording z-score array
    zscore = sess_data.zscore.values
    event_per_min, peak_amp, prom, dist, thresh, peak_width = FP_peak_analysis(zscore, samples_sec)