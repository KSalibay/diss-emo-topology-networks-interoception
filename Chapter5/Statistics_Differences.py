#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 01:19:30 2022

@author: ksalibay
This script is designed to compare the difference between two conditions (e.g. systole and diastole) in a given frequency band.
"""

import numpy as np
from numpy import load
import scipy.io
#Specify your subject, subject directory, labels for phase, phase2, and frequency band
subject = 's1'
subject_dir = 's1'
phase = 'systole'
phase2 = 'diastole'
band = 'alpha'

#Don't forget to define your FDR threshold
threshold = 0.01
#%%
fname_data_sys = str('/N/slate/ksalibay/DataICM2019/'+subject_dir+'_data/'+band+'_'+phase+'/NPYs/'+subject+phase+band+'.npy')
fname_data_dia = str('/N/slate/ksalibay/DataICM2019/'+subject_dir+'_data/'+band+'_'+phase2+'/NPYs/'+subject+phase2+band+'.npy')
sys_data = load(fname_data_sys)
dia_data = load(fname_data_dia)

##%%
#Create a difference array
diff_data = sys_data - dia_data

#%%
#Create an array of matrices which are differences between nulls
diff_nulls = []
subject2 = 's1'
for i in range(1000):
    fname_sys = str('/N/slate/ksalibay/DataICM2019/'+subject_dir+'_nulls/'+band
                    +'_'+phase+'/NPYs/'+subject2+'nounknown_con_'+phase+band+'onlycon'+'aparc2009'+'env'+'null'+str(i)+'.npy')
    fname_dia = str('/N/slate/ksalibay/DataICM2019/'+subject_dir+'_nulls/'
                    +band+'_'+phase2+'/NPYs/'+subject+'nounknown_con_'+phase2+band+'onlycon'+'aparc2009'+'env'+'null'+str(i)+'.npy')
    null_sys = load(fname_sys)
    null_dia = load(fname_dia)
    diff_temp = null_sys - null_dia
    diff_nulls.append(diff_temp)
    
#%%
#Take the lower triangle of the difference arrays in both the data and the nulls
lt_ind = np.tril_indices(148, k=-1)
lt_data = diff_data[lt_ind]

##%%
lt_nulls = []
for a in diff_nulls:
    lt_temp = a[lt_ind]
    lt_nulls.append(lt_temp)
lt_nulls = [l.tolist() for l in lt_nulls]   
##%%
nodes_nulls = []
for each in range(len(lt_nulls[0])):
    nodes_temp = []
    for k in lt_nulls:
        nodes_temp.append(k[each])
    nodes_nulls.append(nodes_temp)

#%%
#Let's create a "mother of all distributions" array which will have NxN rows,
#where each row is a distribution of the data (at index 0) and values in each of the null iterations

tot_lt = []
for eachvalue in range(len(lt_data)):
    tot_lt = nodes_nulls.copy()
    tot_lt[eachvalue].insert(0, lt_data[eachvalue])
    
##%%

import scipy.stats as stats

data_stats = []
for eachnode in range(len(tot_lt)):
    array_temp = np.array(tot_lt[eachnode])
    stats_overall = stats.zscore(array_temp, ddof = 1)
    data_stats.append(stats_overall[0])

abs_data_stats = []
for value in range(len(data_stats)):
    abs_data_stats.append(abs(data_stats[value]))
    
#%%
#Let's correct our t-test values with FDR
#Genovese, C. R., Lazar, N. A., & Nichols, T. (2002). 
#Thresholding of statistical maps in functional neuroimaging using the false discovery rate. Neuroimage, 15(4), 870-878.

from mne import stats as mstats

full_ps = np.zeros((len(abs_data_stats)))
for m in range(len(abs_data_stats)):
    full_ps[m] = scipy.stats.norm.sf(abs_data_stats[m])*2

rej_full, corr_ps_full = mstats.fdr_correction(full_ps, alpha = threshold)

#%%
ind_pass_fdr = [x for x in range(len(rej_full)) if rej_full[x] == True]

#%%
#Let's correct our t-test values with FDR, but this time with threshold = 0.05
#Genovese, C. R., Lazar, N. A., & Nichols, T. (2002). 
#Thresholding of statistical maps in functional neuroimaging using the false discovery rate. Neuroimage, 15(4), 870-878.

from mne import stats as mstats

full_ps = np.zeros((len(abs_data_stats)))
for m in range(len(abs_data_stats)):
    full_ps[m] = scipy.stats.norm.sf(abs_data_stats[m])*2

rej_full2, corr_ps_full2 = mstats.fdr_correction(full_ps, alpha = 0.05)

#%%
ind_pass_fdr2 = [x for x in range(len(rej_full2)) if rej_full2[x] == True]
#%%
from scipy import stats

full_ps = np.zeros((len(abs_data_stats)))
for m in range(len(abs_data_stats)):
    full_ps[m] = scipy.stats.norm.sf(abs_data_stats[m])*2

##%%

indices = []

for z in range(len(full_ps)):
    if full_ps[z] < threshold:
        indices.append(z)

##%%
#In this cell, we are creating the equivalent of our lower triangle matrix to
#be able to correlate numeric indices of our statistically "significant" z-scores with
#anatomical connections.
import pickle
with open("/N/slate/ksalibay/DataICM2019/aparc2009s_labels.txt", "rb") as fp:
    labels = pickle.load(fp)
        
labels_df = np.zeros((148,148), dtype='U256')

for i in range (0,148):
    for j in range (0,148):
        labels_df[i,j] = labels[i]+"-TO-"+labels[j]

labels_tril_ind = np.tril_indices(148, k=-1)
labels_tril = labels_df[labels_tril_ind]
del labels_df

#%%
#Save the data which are statistically significant (with or without FWR correction)
result_df = np.zeros((len(indices),3),dtype='U256')

for k in range(len(indices)):
    result_df[k,0] = abs_data_stats[indices[k]]
    result_df[k,1] = full_ps[indices[k]]
    result_df[k,2] = labels_tril[indices[k]]
    
np.savetxt('/N/slate/ksalibay/DataICM2019/'+subject_dir+'_data/'+band+'_diff_stats.csv',result_df, fmt='%s', delimiter=',')
np.savetxt('/N/slate/ksalibay/DataICM2019/'+subject_dir+'_data/'+band+'total_dists.csv',tot_lt, delimiter=',')
#%%

#Reality check time: let's plot these frequency distributions (at least a couple) to see if they are remotely Gaussian...
#Rerun to generate and save figures as many times as you wish.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

rand = np.random.randint(1, high=1000)
dist1 = tot_lt[rand]

n_bins = 40
fig, ax = plt.subplots(1, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
ax.hist(dist1, bins=n_bins)
plt.axvline(tot_lt[rand][0], color='k', linestyle='dashed', linewidth=1)

fig.savefig(fname='/N/slate/ksalibay/DataICM2019/'+subject_dir+'_data/'+subject+phase+band+str(rand)+'dist.png')