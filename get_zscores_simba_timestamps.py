# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:04:50 2023

@author: zweifeladmin
"""

"""create 2D arrays of GCAMP and ISOS values around each TTL timestamp based on defined time window 
and calculate z-scores based on defined baseline period and method""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_zscores_simba_timestamps(sess_data, sess_behav_data, sec_before_TTL, sec_after_TTL, sample_rate, vid_trim_sec, bout_thresh):
    # make session timestamp zscore window df
    print('processing FP data', sess_data.head(1), '\n')
    print('processing behav data', sess_behav_data.head(1), '\n')
    timestamps = sess_behav_data[['Event', 'Start_time']]
    timestamps['Start_time'] = timestamps.Start_time + vid_trim_sec
    #group timestamp types into bouts (use get bouts code) then use all bout onset times as 'timestamps'
    timestamps_df = pd.DataFrame()
    for t in timestamps.Event.unique():
        mouse_behav_times = timestamps[timestamps.Event==t]
        times_diff = np.diff(mouse_behav_times.Start_time)
        times_diff_filt = np.where(times_diff >= bout_thresh)[0]
        mouse_behav_times_filt = mouse_behav_times.iloc[times_diff_filt]
        behav_timestamps_df = mouse_behav_times_filt
        timestamps_df = pd.concat([timestamps_df, mouse_behav_times_filt])
    #align filtered timestamps to zscore array and make zscore windows df
    zscore_win_dict = {}
    for i in timestamps.values:
        behav = i[0]
        timestamp = i[1]
        FP_time = sess_data.iloc[(sess_data['time'] - timestamp).abs().argsort()[:1]].time.values[0]
        # print('closest time sample in FP zscore trace to behav timestamp: ', FP_time)
        FP_time_index =  sess_data[sess_data.time == FP_time].index.values[0] #index of timestamp in FP df
        # sample_rate = sess_data.index.max() / (sess_data.time.iloc[-1] - sess_data.time.iloc[0])
        start_ind = round(FP_time_index - (sec_before_TTL * sample_rate))
        end_ind = round(FP_time_index + (sec_after_TTL*sample_rate))
        zscore_win =sess_data.loc[start_ind:end_ind].zscore.values
        zscore_win_dict[i[1]] = [i[0], zscore_win]
    zscore_win_df = pd.DataFrame.from_dict(zscore_win_dict, orient = 'columns').T.reset_index()
    zscore_win_df.columns = ['timestamp', 'behav', 'zscore_win']
    
    return(zscore_win_df)