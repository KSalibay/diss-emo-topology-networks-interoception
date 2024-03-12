#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:13:41 2020

@author: Kami Salibayeva

This script takes in EKG data plucked out of FIF-recordings (presumed to be recorded along with M/EEG data on an
                                                             Elekta Neuromag system)
and creates a NumPy array of events corresponding to the start of systolic and diastolic cardiac phases.
It then applies the trapezoidal algorithm to detect the end of the t-wave and the start of diastole.
Utilizes heartpy and numpy.
"""

import mne
import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt

#Load your data here. 
#Define the location of your data and the participant ID.
data_folder = '/N/slate/ksalibay/DataICM2019/output_may23/'
part_id = 3

#%% Load from raw
raw = mne.io.read_raw_fif(data_folder+'S'+str(part_id)+'concatenatedfilt1-40_tsss.fif')

#Pluck out the EKG data based on the channel name
ecg_channel = 'BIO003'
my_ecg = raw[ecg_channel,:]

#The output of above needs further work.
hpdata = my_ecg[0]
conv = np.array(hpdata)

#Back up the ECG data as a .csv
np.savetxt(X=conv.T, fname=data_folder+'S'+str(part_id)+'ECG_data.csv')

#%% Load from csv
#Load your data here if you already have it in .csv format.
hpdata = hp.get_data(data_folder+'S'+str(part_id)+'ECG_data.csv')

#%%
#Visualize the data as is. It will likely have slow drifts.
sample_rate = raw.info['sfreq']
#sample_rate = 1000
plt.figure(figsize=(12,4))
plt.plot(hpdata)
plt.show()

#%%
#Remove baseline wander with HeartPy! Play around with the cutoff value if the wander is not completely eliminated.
hpdata1 = hp.remove_baseline_wander(hpdata, sample_rate, cutoff=0.05)
hpdata1 = hpdata1.squeeze()
plt.figure(figsize=(12,4))
plt.plot(hpdata1)
plt.show()

#%%
#You may want to resample your data. If so, you will need scipy.signal module installed.
#Uncomment the lines below if so.

# from scipy.signal import resample
# scaling_factor = 4
# new_sample_rate = len(hpdata1) * scaling_factor
# resampled_signal = resample(hpdata1, new_sample_rate)
# wd, m = hp.process(hp.scale_data(resampled_signal), new_sample_rate)

#Otherwise (if we already have a high sampling rate of >=1000 Hz), we may proceed as is:
wd, m = hp.process(hp.scale_data(hpdata1), sample_rate)
plt.figure(figsize=(12,4))
hp.plotter(wd, m)

#Look at the output here to detect if something looks weird!
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
  
#%%
#These variables will be needed for the algorithm to wor — just a shortcut for the list of R-peaks,
#time points at each successive pair of R-peaks, and the heartrate itself
R_list = wd['peaklist']
RR_ind = wd['RR_indices']
hr = wd['hr']

#%% DEFINE TRAPEZOIDAL

def trapez_area(xm, ym, xseq, yseq, xr):
    
    '''
    To segment the cardiac cycle into systole and diastole, we computed the 
    trial-specific phases based on cardio-mechanical events related to the ECG 
    trace. 
    The ventricular systolic phase (further referred as "systole") was defined 
    as a time between R peak of the QRS complex and the t-wave end, while 
    diastole as the remaining part of the RR interval. The trapez area algorithm 
    was applied to encode the t-wave end in each trial. 
    First, the t-peak was located as a local maximum within the 
    _physiologically plausible_ interval after the R peak containing the t-wave. 
    Subsequently, the algorithm computed a series of trapezes along the 
    descending part of the t-wave signal, defining the point at which the 
    trapezium's area gets maximal as the t-wave end.

    Function for trapezoidal area computation in t-wave end detection. 
    Adapted from Esra Al & Pawel Motyka, 2020.
    Python version: Kami Salibayeva, 2020.
    '''
    a = []
    for i in range(len(xseq)-1):
        atemp = 0.5 * (ym - yseq[i]) * ((2*xr) - xseq[i] - xm)
        a.append(atemp)
    amax = max(a)
    x_tend = a.index(amax)+xm-1
    return x_tend

#%% RUN TWAVE ALG
#This cell does the magic t-wave end detection (which is presumed to be the start of diastole).
#The values here are quoted from Esra Al and Pawel Motyka, 2020.
fs = sample_rate
twave2 = []
t_end = []
t_val = []
t_maxes = []

for i in range(len(RR_ind)-1):
    ecgpos1 = RR_ind[i][0]
    RR_interval = RR_ind[i][1] - RR_ind[i][0]
    ecgpos2 = int(ecgpos1 + RR_interval - 150*(fs / 2500))
    twave = hr[(ecgpos1+int(fs*(350/2500))):ecgpos2]
    tmax = max(twave[0:int(((RR_interval-350*(fs/2500))/3))])
    tmaxpos = (twave == tmax).nonzero()[0][0]
    twave2 = twave[tmaxpos:len(twave)]
    t_maxes.append(tmaxpos)
    # Determine a point called xm located in the segment after the T peak, 
    # which has a minimum value in the first derivative. The algoritm searces 
    # for xm in a 120 ms time window startng from tmax. In case twave2 does not
    # contain 0.12*fs data points, it searches only until the last point of twave2.
    dp = 0.12*fs # 0.12 s in data points
    if dp > len(twave2):
        xm = (np.diff(twave2[2:])==min(np.diff(twave2[2:]))).nonzero() 
    else:
        xm = (np.diff(twave2[1:int(dp)])==min(np.diff(twave2[1:int(dp)]))).nonzero()
    
    xm = int(xm[0])
    ym = twave2[xm]
    
    # determine a point xr which is supposed t happen after tend.
    xr = int(fs*0.15+xm)
    
    # make a vector starting from xm and goes until xr
    xseq = list(range(xm,xr))
    yseq = list(twave2[xm:xr])
    if len(xseq) != len(yseq):
        t_end.append(0)
        t_val.append(0)
    else:
        tend = trapez_area(xm, ym, xseq, yseq, xr)
        t_end.append(tend)
        t_val.append(twave2[tend])

#%%
#Here we just create lists of values to aid in plotting the t-wave ends in green and ends of diastole in purple.
t_abs = []
for i in range(0,len(t_end)):
    absolute = t_end[i] + R_list[i] + t_maxes[i] + int(fs*(350/2500))
    t_abs.append(absolute)

d_end = []
d_val = []

for i in range(0,len(t_abs)):
    dend = t_abs[i] + (t_abs[i] - R_list[i])
    d_end.append(dend)
    d_val.append(hr[dend])

#Plot the data to make sure the algorithm worked!
plt.figure(figsize=(12,4))
plt = hp.plotter(wd, m)

plt.scatter(t_abs, t_val, marker='d', c='green', label='t-wave ends')
plt.scatter(d_end, d_val, marker='d', c='purple')

#%%
#The following few cells massage the data to make sure that we have suitable lengths of systole and diastole for our
#analysis of epoched data later. First, we create an array with just the R-peaks (start of systole), end of t-wave 
#(start of diastole), and end of diastole (determined by the length of systole, so that diastole is of equal length).
#It also includes an offset that is plucked out of the original FIF file.
R_list_one = R_list[:-2]
epoching = np.empty((len(R_list_one)*3))
epoching[::3] = R_list_one
epoching[1::3] = t_abs
epoching[2::3] = d_end
offset = raw.first_samp
epoching = [t + offset for t in epoching]
events = np.array(epoching)
zeros = np.zeros(len(events))
fourth_column = np.empty(len(events))
fourth_column[::3] = 1
fourth_column[1::3] = 2
fourth_column[2::3] = 3

events = np.vstack((events,zeros))
events = np.vstack((events,fourth_column))

events = events.T
events = events.astype(int)
#We save the event here.
np.savetxt(X=events, fname=data_folder+'S'+str(part_id)+'ECG_events3.csv')

#%%
#However, we also want to make sure that all of our epochs are of equal length. In order to do that,
#we look at figures of distributions of lengths of our epochs and determine a suitable minimum length.
#Let's start by creating the distributions for our intervals — both for systole/diastole durations and for the durations of intervals
#between them.

import matplotlib.pyplot as plt
import numpy as np

data_folder = '/N/slate/ksalibay/DataICM2019/output_may23/'
raw_fname = 'S'+str(part_id)+'ECG_events3.csv'

timepoints = np.genfromtxt(data_folder + raw_fname, delimiter=' ')

#%%
syst_durations = []
int_durations = []
for row in np.arange(0,len(timepoints)):
    if timepoints[row,2] == 2.:
        syst_durations.append(timepoints[row,0] - timepoints[row-1,0])
    elif timepoints[row,2] == 3.:
        int_durations.append(timepoints[row+1,0] - timepoints[row,0])

#%%

np.savetxt(X = syst_durations, fname = data_folder+'S'+str(part_id)+'ECG_events3_syst_durations.csv')
np.savetxt(X = int_durations, fname = data_folder+'S'+str(part_id)+'ECG_events3_int_durations.csv')

#%%
#figure()
dur_syst = np.genfromtxt(data_folder+'S'+str(part_id)+'ECG_events3_syst_durations.csv', skip_header=True)

n, bins, patches = plt.hist(x=dur_syst, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Duration of systolic/diastolic phases')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()

#%%
#figure()
dur_int = np.genfromtxt(data_folder+'S'+str(part_id)+'ECG_events3_int_durations.csv', skip_header=True)

n, bins, patches = plt.hist(x=dur_int, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Duration of intervals between diastole/systole')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()

#%%

"""
=========
Look at the disribution, decide on the suitable minimal duration of a single phase, remove outliers.
=========
"""

events_preproc = np.genfromtxt(data_folder+'S'+str(part_id)+'ECG_events3.csv', delimiter=' ')

events_fourth = np.zeros(len(events_preproc))
events_fourth[::3] = syst_durations
events_fourth[1::3] = syst_durations

events_preproc1 = np.column_stack((events_preproc, events_fourth))
#Determine the minimal length of systole here — we advocate for the 5th or 10th percentile value to avoid overlap
#between the maximal number of systolic and diastolic epochs when applying this to the original FIFF file.
min_length = 325
events_preproc1 = events_preproc1[np.logical_or
                                (np.logical_and
                                 (events_preproc1[:,2]!=3,events_preproc1[:,3]>=min_length),events_preproc1[:,2]==3)]

events_final = np.delete(events_preproc1,3,1)
#%%

np.savetxt(X=events_final, fname=data_folder+'S'+str(part_id)+'ECG_events_no_out.csv')
