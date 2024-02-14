# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:35:31 2023

This script reads in a dataframe with zscore windows for behavior timestamps and 
plots FP traces according to specified groupings. 

@author: Jordan Elum
"""

#%% import libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
import glob

#%% read in pandas dataframe containing zscore windows and info columns 
task = 'group'
df_path = input('Enter the absolute path to the directory in which the z-score window dataframe is stored (no quotes):')
filecsv = glob.glob(df_path + "/*_df.csv")[0]
filenpy = glob.glob(df_path + "/*_df.npy")[0]
columns = pd.read_csv(filecsv, nrows=0).columns.tolist()
data = np.load(filenpy, allow_pickle=True)
allmice_df = pd.DataFrame(data,columns=columns)

#%% directory to save figures
path = input('Enter the absolute path of the directory in which the output results should be saved (no quotes): ')  
out_paths = [os.path.join(path + '\\figures')]
for out in out_paths: 
    if not os.path.exists(out):
        os.makedirs(out)
fig_save_path = out_paths[0]    

#%% plotting custom x ticks and group colors, time array for plotting mean traces  
task = allmice_df.task.unique()
samples = len(allmice_df['zscore_win'].iloc[0])
window_sec = 15
samples_sec = samples/window_sec
x_intervals_sec = 5
x_ticks = list(np.arange(0, samples+1, (samples/window_sec)*x_intervals_sec)) 
x_labels = ['-5', '0', '5', '10']
time = np.arange(0, allmice_df['zscore_win'].iloc[0].size, 1)
color_dict = {'pfc-il-vglut1': 'black', 'msh-vgat':'red'} #choose color dictionary
sec_sample_dict = {'-5s': samples_sec*0, '-3s': samples_sec*2, '0s': samples_sec*5, '3s': samples_sec*8, '6s': samples_sec*11}
# choose b/t plotting entire z-score window trace or shorter window duration
plot_window = (0, int(window_sec*samples_sec)) #default to plot the z-score whole window duration
# sec_to_plot = (3,8) #option to plot only a shorter window in sec (i.e. use (3,8) to plot only 3 s before to 5 s after the timestamp) 
# plot_window = (int(sec_to_plot[0] * samples_sec), int(sec_to_plot[1] *samples_sec)) #option to plot only a shorter window 
behav_list = allmice_df.behav.unique().tolist() #make list to plot traces for all behavior types in the dataframe
group_list = allmice_df.group.unique()

#%% exclude mice by mouseID if needed
exclude_mice = []
exclude_ind = allmice_df[allmice_df['mouseID'].isin(exclude_mice)].index
allmice_df.drop(index=exclude_ind,inplace=True)

#%% mouse by mouse plotting mean trace for each behav type: (1 mean trace/mouse/behav) and save in svg and png formats
for i in allmice_df['mouseID'].unique():
    for j in behav_list:
        plt.figure(figsize=(9,4), dpi = 600)
        count = allmice_df[(allmice_df['mouseID'] == i) & (allmice_df['behav'] == j)]['zscore_win'].count()
        if count != 0:        
            plot = np.mean(allmice_df[(allmice_df['mouseID'] == i) & (allmice_df['behav'] == j)]['zscore_win'])
            block = np.array2string(np.unique(allmice_df[(allmice_df['mouseID'] == i) & (allmice_df['behav'] == j)]['block'].values))
            cohort = np.array2string(np.unique(allmice_df[(allmice_df['mouseID'] == i) & (allmice_df['behav'] == j)]['cohort'].values))
            plt.title(str(i)+'_'+str(j) + '# epochs: ' + str(count) + ' ' + str(block) + ' ' + str(cohort))
            plt.axvline(x=sec_sample_dict['0s'], alpha=0.9, color='k', linewidth=1.2,ls=':')
            plt.plot(time,plot)
            plt.legend()
            # plt.ylim(-1,3.2)
            plt.xlabel('Time (s)')
            plt.ylabel('Z-score')
            plt.xticks(ticks=x_ticks, labels=x_labels)
            # plt.show() 
        else:
            print('no timestamps for ',i,j,task)
plt.savefig(fig_save_path + '\\' + str(allmice_df.task.unique().tolist()) + '_' + str(i) + '_' + str(j) + '_' + str(block) +'.svg',  transparent=True, bbox_inches='tight')
plt.savefig(fig_save_path + '\\' + str(allmice_df.task.unique().tolist()) + '_' + str(j) + '_' + str(block) + '.png', bbox_inches='tight')

#%% average zscore windows by mouseID, then plot group means for each behavior type (1 plot per behav per group)
plot_df = allmice_df[['task','group','mouseID','behav','timestamp','zscore_win']] #subset df 
plot_means_df = plot_df.groupby(['task','behav','group','mouseID'])['zscore_win'].apply(np.mean).reset_index()

#%% plot mean trace according to above grouping (with sem shaded if n > 1) and save in svg and png formats
grouping = plot_means_df # a groupby df containing a time-series array column to plot from
for b in behav_list:
    bsln='zscore_win'
    for g in group_list:
        plt.figure(figsize=(8,6),dpi=1200) #omis
        group_mean = np.mean(grouping[(grouping.behav==b) & (grouping.group==g)][bsln])
        group_sem = sem(list(grouping[(grouping.behav==b) & (grouping.group==g)][bsln]))
        # group_mean = group_mean - np.mean(group_mean[0:203]) # option to subtact a pre timestamp baseline
        plt.plot(time[plot_window[0]:plot_window[1]],group_mean[plot_window[0]:plot_window[1]], linewidth=2.3, label=b,color=color_dict[g])
        plt.fill_between(time[plot_window[0]:plot_window[1]],group_mean[plot_window[0]:plot_window[1]]+group_sem[plot_window[0]:plot_window[1]],group_mean[plot_window[0]:plot_window[1]]-group_sem[plot_window[0]:plot_window[1]], alpha=0.15,color=color_dict[g])
        plt.axvline(x=sec_sample_dict['0s'], alpha=0.9, color='k', linewidth=1.2,ls=':')
        plt.title('Task: ' + str(task) + '\n' + 'Behavior: ' + str(b) + '\n' + str(g),fontsize=19)
        # plt.legend(loc='upper right',fontsize=16)
        plt.xlabel('Time (s)',fontsize=19)
        plt.ylabel('z-score',fontsize=19)
        plt.xticks(ticks=x_ticks, labels=x_labels,fontsize=17)
        plt.yticks(fontsize=17)
        # yrange = (-.8,2.4) 
        # plt.ylim(yrange[0], yrange[1])
        plt.margins(x=0)
        plt.box(on=False)
plt.savefig(fig_save_path + '\\' + str(allmice_df.task.unique().tolist()) + '_' + str(b) + '_' + str(g) + '_mean_trace'  +'.svg',  transparent=True, bbox_inches='tight')
plt.savefig(fig_save_path + '\\' + str(allmice_df.task.unique().tolist()) + '_' + str(b) + '_' + str(g) + '_mean_trace' +'.png', bbox_inches='tight')