# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:58:42 2024

This script reads in raw photometry data and raw behavior timestamp data. Then baseline corrects and zscores the photometry data.
Then reformats the behavior timestamp data. Then aligns the behavior timestamps 'zero' time with the zscore 'zero' time index.
Then makes a pandas dataframe with zscore peri-event time windows for all timestamps. 

@author: Jordan Elum
"""
#%% import libraries and functions
import os
os.chdir(input('Enter the absolute path of the directory in which the code is stored (no quotes): ')) #replace this with the path to your code directory
import glob
import pandas as pd
import numpy as np
import math as math
import csv
from get_zscores_simba_timestamps import get_zscores_simba_timestamps
from get_sess_zscore import get_sess_zscore

#%% find FP data and behavior timestamp data files and add filenames to lists
input_data_path = input('Enter the absolute path of the directory in which the data are stored (no quotes): ')
FP_raw_files, FP_info_files, behav_files  = [], [], []
FP_raw_files = glob.glob(input_data_path + "/*_NonEpochRawData*", recursive = True)  
FP_info_files = glob.glob(input_data_path + "/*_session_info_mouseID.csv", recursive = True)  
behav_files = glob.glob(input_data_path + "/*detailed_bout_data_summary*", recursive = True)  
groups_file = glob.glob(input_data_path + "/*mouse_groups*", recursive = True)  
print('\n', 'All files found: ', list(zip(FP_info_files, FP_raw_files, behav_files, groups_file)), '\n')

#%% set task name variable, make output directories
task_name = 'group' #task to analyze
save_path = input('Enter the absolute path of the directory in which the output results should be saved (no quotes): ')
out_paths = [os.path.join(save_path, 'dataframes', task_name), os.path.join(save_path, 'session_FP_data', task_name)]
for out in out_paths: 
    if not os.path.exists(out):
        print('making new directory: ', out, '\n')
        os.makedirs(out)
    else:
        print('directories for this analysis run already exists: ', out, '\n')  
dataframe_save_path, save_path_traces  = out_paths[0], out_paths[1] #set directory to save dataframe and trace plots/data to

#%% create dictionary to assign mouseIDs to experimental groups
groups = pd.read_csv(groups_file[0], header=0)
group_list, group_dict = [], {}
group_dict = dict(zip([i[0:4] for i in groups.melt().dropna()['value'].astype(str).values], groups.melt().dropna()['variable'].astype(str).values))
group_list = groups.to_dict(orient='list').keys()
print('groups_mice dictionary: ', group_dict, '\n')

#%% input sample rate, downsample factor, input time window around TTL for z-score series, and baseline period
sample_rate_raw = 101.72526245117187 #this is the sample rate after downsampling from ~1017.25 Hz to ~101.725 Hz at block data import
downsample_factor = 10 # this is the factor by which you want to further downsample (i.e. 10 --> to go from 101.725 Hz to 10.17 Hz)
sec_before_TTL = 10 #how many seconds before timestamp to epoch z-scores
sec_after_TTL = 20 #how many seconds after timestamp to epoch z-scores
trim_samples = int(60*sample_rate_raw) #seconds (in samples) to trim from recording start to improve baseline correction
mov_win = 10 #window size in samples for moving window mean in downsampling
zscore_method = ['smooth using a zero phase linear digital filter , downsample to ~10 Hz using a moving window mean, '
                 'fit 405 signal to 470 signal using least-squares regression to calculate fitted control signal, ' 
                  'calculate dff using ((470 signal - fitted control signal) / fitted control signal), '
                  'fit and subtract an exponential to the dff signal to remove baseline drift not captured in the 405 fit, '
                  'zscore this subracted signal using ((dff- mean(dff)) / std(dff))']
# save analysis details to csv (saved in the '/output/' directory)
analysis_details = pd.DataFrame({'mov_win_size': mov_win, 'zscore_method': zscore_method, 'samples_sec':sample_rate_raw, 'trim_samples':trim_samples})
analysis_details.to_csv(save_path + '\\' + task_name + '_analysis_details.csv', index=False)
#%% baseline correct and z-score raw FP recordings, plot raw signals, smoothed/downsampled signals, fit lines, dff, zscore
#   one csv and three png files should be generated and saved in the '/output/traces/task' directory for each input FP data file
allmice_df = pd.DataFrame()
for filename in FP_raw_files:
    data = pd.read_csv(filename, header = 3, dtype = float).dropna(how='all', axis='columns')
    info = pd.read_csv(filename, nrows=1).dropna(how='all', axis='columns')
    print('Processing raw file: ', filename, '\n')
    session_time, sess_dff, sess_zscore, info = get_sess_zscore(data, info, downsample_factor, save_path_traces, mov_win, trim_samples, sample_rate_raw)
    
#%% choose time window around behavior timestamps to epoch and time duration threshold to group timestamps into bouts
sec_before_TTL = 5 #seconds
sec_after_TTL = 10 #seconds
bout_thresh = 10 #seconds    
sample_rate = sample_rate_raw/downsample_factor
zscore_win_len = 154 #expected zscore window length in frames

#%% Read in preprocessed dff/zscore files and align timestamps to zscore traces
zscores_path = save_path_traces
zscore_files = []
zscore_files = glob.glob(zscores_path + "/**/*_sess_dff_zscore.csv", recursive = True)  
print('\n', 'Processed z-scores filenames list:\n', *zscore_files, sep = '\n')

allmice_df = pd.DataFrame()
for data_file in zscore_files:
    sess_data = pd.read_csv(data_file)
    block = data_file.split('\\')[-1].split('_')[2]
    #read in matching behavior timestamp data and matching session info data files
    for behav_file in behav_files:
        if block in behav_file:
            sess_behav_data = pd.read_csv(behav_file) 
        if 'sess_behav_data' not in locals():
            print('Cannot find behavior timestamp data file matching this FP recording file')
    for info_file in FP_info_files:
        if block in info_file:
            sess_info_data = pd.read_csv(info_file) 
        if 'sess_info_data' not in locals():
            print('Cannot find session info data file matching this FP recording file')       
    vid_trim = behav_file.split('\\')[-1].split('_')[-8], behav_file.split('\\')[-1].split('_')[-7] #get video start trim time from behav filename
    vid_trim_sec = (int(vid_trim[0].strip('m')) * 60) + int(vid_trim[1].strip('s'))
    #make dataframe with info columns and zscore column
    print('\n', 'Processing file: ', data_file, '\n')
    zscore_df = get_zscores_simba_timestamps(sess_data, sess_behav_data, sec_before_TTL, sec_after_TTL, sample_rate, vid_trim_sec, bout_thresh)
    info_columns = [] #get session info rows and create arrays to match length of zscore df so that all trials have session info aligned
    for i in sess_info_data.columns:
        i = np.array([sess_info_data[i]]*len(zscore_df))
        info_columns.append(i)  
    info_stack = np.hstack(info_columns)
    label_columns = sess_info_data.columns
    labels_df = pd.DataFrame(info_stack, columns=label_columns)
    mouse_session_df = pd.concat([labels_df, zscore_df], axis='columns') #merge z_score_df with info with filled in info rows to same # as zscore_df 
    print('\n', 'Finished adding session/mouse to dataframe: ', sess_info_data.block.values[0], sess_info_data.mouseID.values[0], '\n')
    allmice_df = pd.concat([allmice_df, mouse_session_df], ignore_index=True)
#insert 'group' column into dataframe and remove nan rows and short zscore window rows
allmice_df['mouseID'] = allmice_df['mouseID'].apply(str)
allmice_df.insert(loc=9, column = 'group', value=allmice_df['mouseID'].map(group_dict)) 
print('\n', 'Finished adding sessions to dataframe allmice_df:', allmice_df['block'].unique(), '\n')

short_windows = [i for i,j in zip(allmice_df.zscore_win.index, allmice_df.zscore_win) if len(j) != zscore_win_len]
print('short windows ', short_windows)
allmice_df.drop(index=short_windows,inplace=True) #drop z-score windows smaller than expected size

nan_rows = [i for i,j in zip(allmice_df.zscore_win.index, allmice_df.zscore_win) if np.isnan(j[0]) == True]
print('nan rows index', nan_rows)
allmice_df.drop(index=nan_rows,inplace=True) #drop any dataframe rows with missing data
#confirm that no short z-score windows or 'nan' rows remain in the dataframe
print('nan cols: ', allmice_df.columns[allmice_df.isna().any()].tolist()) 
print('unique z-score window lengths and total number of z-score windows: ', allmice_df['zscore_win'].str.len().value_counts()) # list any short windows        

#%% save dataframe to npy and csv formats and dataframe details to csv (3 files saved in the '/output/dataframes/' directory)
np.save(dataframe_save_path + '\\' + task_name +  '_df.npy', allmice_df) 
allmice_df.to_csv(dataframe_save_path + '\\' + task_name +  '_df.csv', index = False) 
print('dataframe saved as: ', '\n', dataframe_save_path+task_name+'_df.csv')
df_details = pd.DataFrame({'sec_before_TTL': sec_before_TTL, 'sec_after_TTL': sec_after_TTL, 'sample_rate': sample_rate, 
                           'zscore_win_len': zscore_win_len, 'zscore_filenames':zscore_files})
df_details.to_csv(dataframe_save_path + '\\' + task_name + '_df_analysis_details.csv', index=False)
print('dataframe analysis details saved as: ',dataframe_save_path + '\\' + task_name + '_df_analysis_details.csv')