#Script for basic stats on static dims, participant Ns and other demographics

#%%
#==============================================================================
#We have a folder with all the data that passed the cleaning stage.
#Let's pull out all the unique IDs from the filenames and compare them to the IDs stored in log_full
#First, let's import the log_full and put all the pins into pins
from typing import Final
import pandas as pd
import numpy as np

input = '/N/slate/ksalibay/Behavioral_Data/Data_Nov23/'
fname = 'pins_final.csv'
pins = pd.read_csv(input+fname)
#Let's only get column 2 from pins and convert it into a list
pins = pins.iloc[:,1]
pins = pins.tolist()

log_full = pd.read_csv(input+'log_20231021.csv')

#%%
#Now let's pull out all the IDs from filenames
import os
import re

#Let's first get the list of files in the directory that have the mask input+'participant_'+pin+'_row_' and end with '.csv'
files = os.listdir(input)
files_to_check = []
for file in files:
    if re.match('participant_[0-9]+_umap_with_emolabels.csv', file):
        files_to_check.append(file)

#%%
#Now let's get the list of pins from the filenames: pins will be a list of strings from the pattern 'participant_'+pin+'_umap_with_emolabels.csv'
#(we will remove the 'participant_' and '_row_' parts later)

pins_from_filenames = []
for file in files_to_check:
    pins_from_filenames.append(re.findall('participant_[0-9]+_umap_with_emolabels.csv', file)[0])
    pins_from_filenames = [x.replace('participant_', '') for x in pins_from_filenames]
    pins_from_filenames = [x.replace('_umap_with_emolabels.csv', '') for x in pins_from_filenames]

#Finally, let's clean pins_from_filenames to only contain unique values
pins_from_filenames = list(set(pins_from_filenames))
#%%
#And now we can compare pins_from_filenames to pins

difference = len(pins) - len(pins_from_filenames)
N = len(pins_from_filenames)
# %%
#Now let's do some basic stats on the demographic data from log_full
#Let's have the column list handy. The true column titles are stored in the first row of log_full

columns = log_full.iloc[0,:]
columns = list(columns)
# %%
#Let's remove the data that are irrelevant to these computations from log_full
#We don't need columns with indices 22 to 87, 92 to 158, 163 to 229, 234 to 300, and 305 to 371.
#Let's create a new dataframe with only relevant columns

log_stats = log_full.iloc[:,0:22]
log_stats = log_stats.join(log_full.iloc[:,87:92])
log_stats = log_stats.join(log_full.iloc[:,158:163])
log_stats = log_stats.join(log_full.iloc[:,229:234])
log_stats = log_stats.join(log_full.iloc[:,300:305])
log_stats = log_stats.join(log_full.iloc[:,371:])
# %%
#Now let's import the static dims data that contains the demographic data

fname_static = input+'init_test_cleaned.csv'
static = pd.read_csv(fname_static)

#%%
#Let's create a log_static dataframe that will only contain the data from static that correspond to 
# pins_from_filenames that are common between log_stats and static
#Let's make sure that log_stats['ExternalReference'] and static['PIN ID'] are both strings
log_stats_pins = [str(x) for x in log_stats['ExternalReference'] if x != 'nan' and x != 'ExternalReference'][2:]
static_pins = [str(x) for x in static['PIN ID']]
log_static = pd.DataFrame()
for pin in pins_from_filenames:
    if str(pin) in log_stats_pins:
        log_static = log_static.append(static[static['PIN ID'].astype(str) == str(pin)])
# %%
#Now let's create a Series of distributions of values from all the columns of log_static that start with MAIA

maia_columns = [col for col in log_static.columns if 'MAIA' in col]

#%%
#Let's go through every column in maia_columns and create a distribution of values for that column,
#store it separately, and then plot it as a histogram

maia_noticing = log_static[maia_columns[0]].dropna()
#Let's drop the index of maia_noticing, and then convert it to a list
maia_noticing = maia_noticing.reset_index(drop=True)
maia_noticing = maia_noticing.tolist()
#%%
output = '/N/slate/ksalibay/Behavioral_Data/Data_Nov23/figures/'

#Let's import the plotting library
import matplotlib.pyplot as plt

#Now let's plot maia_noticing as a histogram
plt.hist(maia_noticing)
#Let's add the labels and title
#plt.title('MAIA-2 Noticing subscale')
plt.xlabel('MAIA-2 Noticing')
plt.ylabel('Frequency')
#Let's also make the font size of the labels and title larger
plt.rcParams.update({'font.size': 20})
#Let's also save this histogram
plt.savefig(output+'maia_noticing.png')
plt.show()
# %%
#Let's do the same to the rest of the columns in maia_columns
maia_not_distracted = log_static[maia_columns[1]].dropna()
maia_not_distracted = maia_not_distracted.reset_index(drop=True)
maia_not_distracted = maia_not_distracted.tolist()
plt.hist(maia_not_distracted)
#plt.title('MAIA-2 Not Distracted subscale')
plt.xlabel('MAIA-2 Not Distracted')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 18})
plt.savefig(output+'maia_not_distracted.png')
plt.show()
#%%
maia_not_worrying = log_static[maia_columns[2]].dropna()
maia_not_worrying = maia_not_worrying.reset_index(drop=True)
maia_not_worrying = maia_not_worrying.tolist()
plt.hist(maia_not_worrying)
#plt.title('MAIA-2 Not Worrying subscale')
plt.xlabel('MAIA-2 Not Worrying')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 14})
plt.savefig(output+'maia_not_worrying.png')
plt.show()
# %%
maia_attention = log_static[maia_columns[3]].dropna()
maia_attention = maia_attention.reset_index(drop=True)
maia_attention = maia_attention.tolist()
plt.hist(maia_attention)
#plt.title('MAIA-2 Attention Regulation subscale')
plt.xlabel('MAIA-2 Attention Regulation')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 18})
plt.savefig(output+'maia_attention.png')
plt.show()

#%%
maia_awareness = log_static[maia_columns[4]].dropna()
maia_awareness = maia_awareness.reset_index(drop=True)
maia_awareness = maia_awareness.tolist()
plt.hist(maia_awareness)
#plt.title('MAIA-2 Emotional Awareness subscale')
plt.xlabel('MAIA-2 Emotional Awareness')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 14})
plt.savefig(output+'maia_awareness.png')
plt.show()
#%%
maia_selfregulation = log_static[maia_columns[5]].dropna()
maia_selfregulation = maia_selfregulation.reset_index(drop=True)
maia_selfregulation = maia_selfregulation.tolist()
plt.hist(maia_selfregulation)
#plt.title('MAIA-2 Self-Regulation subscale')
plt.xlabel('MAIA-2 Self-Regulation')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 20})
plt.savefig(output+'maia_selfregulation.png')
plt.show()

#%%
maia_bodylistening = log_static[maia_columns[6]].dropna()
maia_bodylistening = maia_bodylistening.reset_index(drop=True)
maia_bodylistening = maia_bodylistening.tolist()
plt.hist(maia_bodylistening)
#plt.title('MAIA-2 Body Listening subscale')
plt.xlabel('MAIA-2 Body Listening')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 14})
plt.savefig(output+'maia_bodylistening.png')
plt.show()
#%%
maia_trust = log_static[maia_columns[7]].dropna()
maia_trust = maia_trust.reset_index(drop=True)
maia_trust = maia_trust.tolist()
plt.hist(maia_trust)
#plt.title('MAIA-2 Trusting subscale')
plt.xlabel('MAIA-2 Trusting')
plt.ylabel('Frequency')
plt.rcParams.update({'font.size': 20})
plt.savefig(output+'maia_trust.png')
plt.show()
#%%
#Let's print out the mean and standard deviation of each of the MAIA subscales and limit the decimal places to 2
print('The mean and standard deviation of the MAIA subscales are as follows: \n Noticing: M = '+str(round(np.mean(maia_noticing),2))+'; SD = '+str(round(np.std(maia_noticing),2))+
        '\n Not Distracted: M = '+str(round(np.mean(maia_not_distracted),2))+'; SD = '+str(round(np.std(maia_not_distracted),2))+
        '\n Not Worrying: M = '+str(round(np.mean(maia_not_worrying),2))+'; SD = '+str(round(np.std(maia_not_worrying),2))+
        '\n Attention Regulation: M = '+str(round(np.mean(maia_attention),2))+'; SD = '+str(round(np.std(maia_attention),2))+
        '\n Emotional Awareness: M = '+str(round(np.mean(maia_awareness),2))+'; SD = '+str(round(np.std(maia_awareness),2))+
        '\n Self-Regulation: M = '+str(round(np.mean(maia_selfregulation),2))+'; SD = '+str(round(np.std(maia_selfregulation),2))+
        '\n Body Listening: M = '+str(round(np.mean(maia_bodylistening),2))+'; SD = '+str(round(np.std(maia_bodylistening),2))+
        '\n Trusting: M = '+str(round(np.mean(maia_trust),2))+'; SD = '+str(round(np.std(maia_trust),2)))
#%%
#Let's also concatenate all the figures into one figure of a suitable size to fit a letter-sized page

fig, axs = plt.subplots(4, 2, figsize=(18, 17))
#Let's also set the padding between the subplots to 0.4
plt.subplots_adjust(hspace=0.4)
axs[0, 0].hist(maia_noticing)
axs[0, 0].set_title('Noticing')
axs[0, 0].set_ylabel('Frequency')
#Let's add ticks at every integer on the x-axis and make the limits run from 0 to 5
axs[0, 0].set_xlim(1,6)
axs[0, 0].set_xticks(np.arange(1,6,1))
axs[0, 1].hist(maia_not_distracted)
axs[0, 1].set_title('Not distracted')
axs[0, 1].set_xlim(0,5)
axs[0, 1].set_xticks(np.arange(1,6,1))
axs[1, 0].hist(maia_not_worrying)
axs[1, 0].set_title('Not Worrying')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_xlim(0,5)
axs[1, 0].set_xticks(np.arange(1,6,1))
axs[1, 1].hist(maia_attention)
axs[1, 1].set_title('Attention Regulation')
axs[1, 1].set_xlim(0,5)
axs[1, 1].set_xticks(np.arange(1,6,1))
axs[2, 0].hist(maia_awareness)
axs[2, 0].set_title('Emotional Awareness')
axs[2, 0].set_ylabel('Frequency')
axs[2, 0].set_xlim(0,5)
axs[2, 0].set_xticks(np.arange(1,6,1))
axs[2, 1].hist(maia_selfregulation)
axs[2, 1].set_title('Self-Regulation')
axs[2, 1].set_xlim(0,5)
axs[2, 1].set_xticks(np.arange(1,6,1))
axs[3, 0].hist(maia_bodylistening)
axs[3, 0].set_title('Body Listening')
axs[3, 0].set_ylabel('Frequency')
axs[3, 0].set_xlim(0,5)
axs[3, 0].set_xticks(np.arange(1,6,1))
axs[3, 1].hist(maia_trust)
axs[3, 1].set_title('Trusting')
axs[3, 1].set_xlim(0,5)
axs[3, 1].set_xticks(np.arange(1,6,1))
plt.savefig(output+'maia_all.png')
plt.show()
# %%
#Now let's do the same to all columns in log_static that start with 'SRIS'
sris_columns = [col for col in log_static.columns if 'SRIS' in col]
#%%
sris_engagement = log_static[sris_columns[0]].dropna()
sris_engagement = sris_engagement.reset_index(drop=True)
sris_engagement = sris_engagement.tolist()
plt.hist(sris_engagement)
plt.title('SRIS Engagement in Self-reflection subscale')
plt.xlabel('SRIS Engagement in Self-reflection')
plt.ylabel('Frequency')
plt.savefig(input+'sris_engagement.png')
plt.show()

sris_need = log_static[sris_columns[1]].dropna()
sris_need = sris_need.reset_index(drop=True)
sris_need = sris_need.tolist()
plt.hist(sris_need)
plt.title('SRIS Need for Self-reflection subscale')
plt.xlabel('SRIS Need for Self-reflection')
plt.ylabel('Frequency')
plt.savefig(input+'sris_need.png')
plt.show()

sris_insight = log_static[sris_columns[2]].dropna()
sris_insight = sris_insight.reset_index(drop=True)
sris_insight = sris_insight.tolist()
plt.hist(sris_insight)
plt.title('SRIS Insight subscale')
plt.xlabel('SRIS Insight')
plt.ylabel('Frequency')
plt.savefig(input+'sris_insight.png')
plt.show()

#%%
#Let's create a figure with all the SRIS subscales
fig, axs = plt.subplots(3, 1, figsize=(6,11))
#Let's also set the padding between the subplots to 0.4
plt.subplots_adjust(hspace=0.4)
#Let's also make the font size of the labels and title larger
plt.rcParams.update({'font.size': 20})
#Let's also make sure that the x-axis runs from 0 to 6 for all subplots
axs[0].hist(sris_engagement)
axs[0].set_title('Engagement in Self-reflection')
axs[0].set_ylabel('Frequency')
axs[0].set_xlim(1,6)
axs[1].hist(sris_need)
axs[1].set_title('Need for Self-reflection')
axs[1].set_ylabel('Frequency')
axs[1].set_xlim(1,6)
axs[2].hist(sris_insight)
axs[2].set_title('Insight')
axs[2].set_ylabel('Frequency')
axs[2].set_xlim(1,6)
plt.savefig(output+'sris_all.png')
plt.show()

#%%
#Let's print out the mean and standard deviation of each of the SRIS subscales and limit the decimal places to 2
print('The mean and standard deviation of the SRIS subscales are as follows: \n Engagement in Self-reflection: M = '+str(round(np.mean(sris_engagement),2))+'; SD = '+str(round(np.std(sris_engagement),2))+
        '\n Need for Self-reflection: M = '+str(round(np.mean(sris_need),2))+'; SD = '+str(round(np.std(sris_need),2))+
        '\n Insight: M = '+str(round(np.mean(sris_insight),2))+'; SD = '+str(round(np.std(sris_insight),2)))
# %%
#Now let's do the same to all columns in log_static that start with 'IRI'

iri_columns = [col for col in log_static.columns if 'IRI' in col]
    
#%%
iri_perspective = log_static[iri_columns[0]].dropna()
iri_perspective = iri_perspective.reset_index(drop=True)
iri_perspective = iri_perspective.tolist()
plt.hist(iri_perspective)
plt.title('IRI Perspective Taking subscale')
plt.xlabel('IRI Perspective Taking')
plt.ylabel('Frequency')
plt.savefig(input+'iri_perspective.png')
plt.show()

iri_fantasy = log_static[iri_columns[1]].dropna()
iri_fantasy = iri_fantasy.reset_index(drop=True)
iri_fantasy = iri_fantasy.tolist()
plt.hist(iri_fantasy)
plt.title('IRI Fantasy subscale')
plt.xlabel('IRI Fantasy')
plt.ylabel('Frequency')
plt.savefig(input+'iri_fantasy.png')
plt.show()

iri_empathic = log_static[iri_columns[2]].dropna()
iri_empathic = iri_empathic.reset_index(drop=True)
iri_empathic = iri_empathic.tolist()
plt.hist(iri_empathic)
plt.title('IRI Empathic Concern subscale')
plt.xlabel('IRI Empathic Concern')
plt.ylabel('Frequency')
plt.savefig(input+'iri_empathic.png')
plt.show()

iri_distress = log_static[iri_columns[3]].dropna()
iri_distress = iri_distress.reset_index(drop=True)
iri_distress = iri_distress.tolist()
plt.hist(iri_distress)
plt.title('IRI Personal Distress subscale')
plt.xlabel('IRI Personal Distress')
plt.ylabel('Frequency')
plt.savefig(input+'iri_distress.png')
plt.show()
#%%
#Let's create a figure with all the IRI subscales on a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(8.5,8.5))
#Let's also set the padding between the subplots to 0.4
plt.subplots_adjust(hspace=0.6, wspace=0.3)
#Let's also make the font size of the labels and title larger
plt.rcParams.update({'font.size': 18})
#Let's also make sure to add a frequency label to the y-axis of the left plots
axs[0, 0].hist(iri_perspective)
axs[0, 0].set_title('Perspective Taking')
axs[0, 0].set_ylabel('Frequency')
#Let's set all the x-axes to run from 1 to 5 and let's add tick marks at every integer
axs[0, 0].set_xlim(1,5)
axs[0, 0].set_xticks(np.arange(1,6,1))
axs[0, 1].hist(iri_fantasy)
axs[0, 1].set_title('Fantasy')
axs[0, 1].set_xlim(1,5)
axs[0, 1].set_xticks(np.arange(1,6,1))
axs[1, 0].hist(iri_empathic)
axs[1, 0].set_title('Empathic Concern')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_xlim(1,5)
axs[1, 0].set_xticks(np.arange(1,6,1))
axs[1, 1].hist(iri_distress)
axs[1, 1].set_title('Personal Distress')
axs[1, 1].set_xlim(1,5)
axs[1, 1].set_xticks(np.arange(1,6,1))
plt.savefig(output+'iri_all.png')
plt.show()

#%%
#Let's create a new figure using this code as a template
## Create a figure with 8 subplots in a 4x2 grid for histograms
# fig_maia, axes_maia = plt.subplots(4, 2, figsize=(8.5, 11))  # Setup for a letter size page

# # Load data, plot histograms, and save the figure
# for path, ax, title in zip(file_paths_maia, axes_maia.flatten(), maia_titles):
#     data = pd.read_csv(path)
#     ax.hist(data.iloc[:, 1], bins='auto', color='blue', alpha=0.7)
#     ax.set_title(title)

# fig_maia.tight_layout()
# fig_maia.savefig('maia_histograms.png', dpi=300)
#Let's read in MAIA-2 data from the csv files and plot them as histograms; the files are in output+name.csv
file_paths_maia = [output+'maia_noticing.csv', output+'maia_not_distracted.csv', output+'maia_not_worrying.csv', output+'maia_attention.csv', output+'maia_awareness.csv', output+'maia_selfregulation.csv', output+'maia_bodylistening.csv', output+'maia_trust.csv']
maia_titles = ['Noticing', 'Not Distracted', 'Not Worrying', 'Attention Regulation', 'Emotional Awareness', 'Self-Regulation', 'Body Listening', 'Trusting']
#%%
fig, axs = plt.subplots(4, 2, figsize=(8.5,11))
for path, ax, title in zip(file_paths_maia, axs.flatten(), maia_titles):
    data = pd.read_csv(path)
    ax.hist(data.iloc[:, 1], bins='auto', color='blue', alpha=0.7)
    ax.set_title(title)
    #Let's reduce the font size of the title to 14
    ax.title.set_fontsize(16)
    #Let's also set the padding between the subplots to 0.4
    plt.subplots_adjust(hspace=0.4)
    ax.set_xlim(1,5)
    ax.set_xticks(np.arange(0,6,1))
    #Let's only leave the y-axis label on the left plots
    if ax in axs[:,0]:
        ax.set_ylabel('Frequency')

fig.savefig(output+'maia_all.png', dpi=300)

#%%
#Let's do that same for IRI
file_paths_iri = [output+'iri_perspective.csv', output+'iri_fantasy.csv', output+'iri_empathic.csv', output+'iri_distress.csv']
iri_titles = ['Perspective Taking', 'Fantasy', 'Empathic Concern', 'Personal Distress']

fig, axs = plt.subplots(2, 2, figsize=(8.5,8.5))
for path, ax, title in zip(file_paths_iri, axs.flatten(), iri_titles):
    data = pd.read_csv(path)
    ax.hist(data.iloc[:, 1], bins='auto', color='blue', alpha=0.7)
    ax.set_title(title)
    #Let's reduce the font size of the title to 14
    ax.title.set_fontsize(16)
    #Let's also set the padding between the subplots to 0.4
    plt.subplots_adjust(hspace=0.4)
    ax.set_xlim(1,5)
    ax.set_xticks(np.arange(1,6,1))
    #Let's only leave the y-axis label on the left plots
    if ax in axs[:,0]:
        ax.set_ylabel('Frequency')

fig.savefig(output+'iri_all.png', dpi=300)

#%%

#Let's also create a Q-Q plot for each of the distributions to test for normality in IRI subscales

import scipy.stats as stats

fig, axs = plt.subplots(2, 2, figsize=(8.5,8.5))
for path, ax, title in zip(file_paths_iri, axs.flatten(), iri_titles):
    data = pd.read_csv(path)
    stats.probplot(data.iloc[:, 1], dist="norm", plot=ax)
    ax.set_title(title)
    #Let's reduce the font size of the title to 14
    ax.title.set_fontsize(16)
    #Let's set the x-axis label font size to 12
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    #Let's also set the padding between the subplots to 0.4
    plt.subplots_adjust(hspace=0.4)
    #Let's only leave the y-axis label on the left plots
    if ax in axs[:,0]:
        ax.set_ylabel('Ordered Values', fontsize=12)

fig.savefig(output+'iri_qq.png', dpi=300)
#%%
#Let's also print out the mean and standard deviation of each of the IRI subscales and limit the decimal places to 2
print('The mean and standard deviation of the IRI subscales are as follows: \n Perspective Taking: M = '+str(round(np.mean(iri_perspective),2))+'; SD = '+str(round(np.std(iri_perspective),2))+
        '\n Fantasy: M = '+str(round(np.mean(iri_fantasy),2))+'; SD = '+str(round(np.std(iri_fantasy),2))+
        '\n Empathic Concern: M = '+str(round(np.mean(iri_empathic),2))+'; SD = '+str(round(np.std(iri_empathic),2))+
        '\n Personal Distress: M = '+str(round(np.mean(iri_distress),2))+'; SD = '+str(round(np.std(iri_distress),2)))

#%%
#Let's also test the distributions separately for each subscale to see if they are normally distributed
#We can use Shapiro-Wilk test and Kolmogorov-Smirnov test for this
#Don't forget to correct for multiple comparisons
import scipy.stats as stats

alpha = 0.05
n_tests = 8
N = len(maia_noticing)
maia_shapiro_wilk = []
maia_kolmogorov_smirnov = []
sris_shapiro_wilk = []
sris_kolmogorov_smirnov = []
iri_shapiro_wilk = []
iri_kolmogorov_smirnov = []

#Let's start with the MAIA subscales
maia_shapiro_wilk.append(stats.shapiro(maia_noticing)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_noticing, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_not_distracted)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_not_distracted, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_not_worrying)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_not_worrying, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_attention)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_attention, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_awareness)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_awareness, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_selfregulation)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_selfregulation, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_bodylistening)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_bodylistening, 'norm')[1])
maia_shapiro_wilk.append(stats.shapiro(maia_trust)[1])
maia_kolmogorov_smirnov.append(stats.kstest(maia_trust, 'norm')[1])


#Now let's print out the results and limit the decimal places to 3
print('The p-values for the Shapiro-Wilk test for normality for the MAIA subscales are as follows: \n Noticing: '+str(round(maia_shapiro_wilk[0],3))+
        '\n Not Distracted: '+str(round(maia_shapiro_wilk[1],3))+
        '\n Not Worrying: '+str(round(maia_shapiro_wilk[2],3))+
        '\n Attention Regulation: '+str(round(maia_shapiro_wilk[3],3))+
        '\n Emotional Awareness: '+str(round(maia_shapiro_wilk[4],3))+
        '\n Self-Regulation: '+str(round(maia_shapiro_wilk[5],3))+
        '\n Body Listening: '+str(round(maia_shapiro_wilk[6],3))+
        '\n Trusting: '+str(round(maia_shapiro_wilk[7],3)))

#Let's print out the results for the Kolmogorov-Smirnov test in scientific notation with 3 decimal places
print('The p-values for the Kolmogorov-Smirnov test for normality for the MAIA subscales are as follows: \n Noticing: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[0]))+
        '\n Not Distracted: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[1]))+
        '\n Not Worrying: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[2]))+
        '\n Attention Regulation: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[3]))+
        '\n Emotional Awareness: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[4]))+
        '\n Self-Regulation: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[5]))+
        '\n Body Listening: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[6]))+
        '\n Trusting: '+str('{:.3e}'.format(maia_kolmogorov_smirnov[7])))


#%%
#Let's save all distributions as csv files
maia_noticing_csv = pd.Series(maia_noticing)
maia_not_distracted_csv = pd.Series(maia_not_distracted)
maia_not_worrying_csv = pd.Series(maia_not_worrying)
maia_attention_csv = pd.Series(maia_attention)
maia_awareness_csv = pd.Series(maia_awareness)
maia_selfregulation_csv = pd.Series(maia_selfregulation)
maia_bodylistening_csv = pd.Series(maia_bodylistening)
maia_trust_csv = pd.Series(maia_trust)

maia_noticing_csv.to_csv(output+'maia_noticing.csv')
maia_not_distracted_csv.to_csv(output+'maia_not_distracted.csv')
maia_not_worrying_csv.to_csv(output+'maia_not_worrying.csv')
maia_attention_csv.to_csv(output+'maia_attention.csv')
maia_awareness_csv.to_csv(output+'maia_awareness.csv')
maia_selfregulation_csv.to_csv(output+'maia_selfregulation.csv')
maia_bodylistening_csv.to_csv(output+'maia_bodylistening.csv')
maia_trust_csv.to_csv(output+'maia_trust.csv')

#%%
#Let's also export SRIS and IRI distributions as csv files
sris_engagement_csv = pd.Series(sris_engagement)
sris_need_csv = pd.Series(sris_need)
sris_insight_csv = pd.Series(sris_insight)

sris_engagement_csv.to_csv(output+'sris_engagement.csv')
sris_need_csv.to_csv(output+'sris_need.csv')
sris_insight_csv.to_csv(output+'sris_insight.csv')

iri_perspective_csv = pd.Series(iri_perspective)
iri_fantasy_csv = pd.Series(iri_fantasy)
iri_empathic_csv = pd.Series(iri_empathic)
iri_distress_csv = pd.Series(iri_distress)

iri_perspective_csv.to_csv(output+'iri_perspective.csv')
iri_fantasy_csv.to_csv(output+'iri_fantasy.csv')
iri_empathic_csv.to_csv(output+'iri_empathic.csv')
iri_distress_csv.to_csv(output+'iri_distress.csv')

#%%
#Now let's import the file init_test_20230911.csv and look at the basic demographics

fname_init = input+'init_test_20230911.csv'
init_full = pd.read_csv(fname_init)
# %%
#The columns we need from init_full are: 'Q9','Q10','Q11', and 'PIN ID'
#Let's create a new dataframe with only these columns
init_demo = init_full[['Q9','Q10','Q11','PIN ID']]

#Let's also cross-reference the PIN IDs in init_demo with pins_from_filenames and drop the rows that don't match
init_demo = init_demo[init_demo['PIN ID'].isin(pins_from_filenames)]
# %%
#Let's count 'Male' vs 'Female' or 'Non-binary' responses in Q9
genders = init_demo['Q9'].value_counts()
print('The gender distribution in our final sample was: '+ str(genders[0]) + ' Male; ' + \
      str(genders[1]) + ' Female; '+str(genders[2])+' Non-binary.')
#Let's also plot the age distribution from Q10
ages = init_demo['Q10'].dropna()
ages = ages.reset_index(drop=True)
ages = ages.tolist()

## %%
#Let's replace values of 20 and 21 with 22 in ages and then calculate the age stats
ages = [x.replace('20', '22') for x in ages]
ages = [x.replace('21', '22') for x in ages]
ages = [int(x) for x in ages]
avg_age = np.mean(ages)
std_age = np.std(ages)
print('The age range was '+str(min(ages))+' to '+str(max(ages))+' years old (M = '+str(avg_age)+', SD = '+str(std_age)+')')
# %%
#Let's plot the age distribution
plt.hist(ages)

plt.savefig(input+'ages.png')
plt.show()
# %%
#Finally, let's bin together all the ethnicities in Q11 that contain 'White' or 'Caucasian',
#all the ones which contain 'Black' or 'African', and then put the rest in a separate frame to sort through manually
tally_white = 0
tally_black = 0
tally_other = 0
other_manual = []
#Let's loop through init_demo['Q11'] and count the number of 'White' and 'Black' responses
ethnicities = init_demo['Q11'].dropna()

for i in ethnicities:
    if 'White' in i or 'Caucasian' in i:
        tally_white += 1
    elif 'Black' in i or 'African' in i:
        tally_black += 1
    else:
        tally_other += 1
        other_manual.append(i)

#%%
#Run this cell to sort the top value of other_manual into 'Black'
tally_black += 1
tally_other -= 1
other_manual = other_manual[1:]
#%%
tally_white += 1
tally_other -= 1
other_manual = other_manual[1:]
#%%
#Run this to leave the top value of other_manual as is in tally_other
other_manual = other_manual[1:]
# %%
tally_native = 0
tally_asian = 0
tally_hispanic = 1
tally_middleeastern = 0
#%%
#Run this cell to sort the top value of other_manual into 'Hispanic'
tally_hispanic += 1
tally_other -= 1
other_manual = other_manual[1:]
#%%
#Run this cell to sort the top value of other_manual into 'Asian'
tally_asian += 1
tally_other -= 1
other_manual = other_manual[1:]

#%%
#Run this cell to sort the top value of other_manual into 'Native American'
tally_native += 1
tally_other -= 1
other_manual = other_manual[1:]
#%%
#Run this cell to sort the top value of other_manual into 'Middle Eastern'
tally_middleeastern += 1
tally_other -= 1
other_manual = other_manual[1:]
# %%
print('Tallies of ethnicities: \n White: '+str(tally_white)+
      '\n Black: '+str(tally_black)+
      '\n Hispanic: '+str(tally_hispanic)+
      '\n Asian: '+str(tally_asian)+
      '\n Native American: '+str(tally_native)+
      '\n Middle Eastern: '+str(tally_middleeastern)+
      '\n Other: '+str(tally_other))
# %%
print('The age range was '+str(min(ages))+' to '+str(max(ages))+' years old. \n The average age was '+str(avg_age)+' years old, with a standard deviation of '+str(std_age)+' years.')
# %%
#Let's change the ages that are 20 and 21 to 22 in init_demo and then calculate the age stats
init_demo['Q10'] = init_demo['Q10'].replace('20', '22')
init_demo['Q10'] = init_demo['Q10'].replace('21', '22')

ages = init_demo['Q10'].dropna()
ages = ages.reset_index(drop=True)
ages = ages.tolist()
ages = [int(x) for x in ages]
avg_age = np.mean(ages)
std_age = np.std(ages)

print('The age range was '+str(min(ages))+' to '+str(max(ages))+' years old. \n The average age was '+str(avg_age)+' years old, with a standard deviation of '+str(std_age)+' years.')
# %%
print('The genders of participants were:\n Male: '+str(genders[0])
      +'\n Female: '+str(genders[1])
      +'\n Non-binary: '+str(genders[2]))
# %%
#Let's save pins_from_filenames as a csv file
pins_from_filenames = pd.Series(pins_from_filenames)
pins_from_filenames.to_csv(input+'pins_final.csv')
# %%
#Let's also take the stats of which ethnicity a PIN ID is associated with and save it as a csv file

ethnicities_and_pins_temp = [x for x in zip(init_demo['PIN ID'], init_demo['Q11'])]
ethnicities_and_pins = []
#%%
num = 0

#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'White'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'White'))
num += 1
#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'Black'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'Black'))
num += 1
#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'Hispanic'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'Hispanic'))
num += 1
#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'Asian'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'Asian'))
num += 1
#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'Native'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'Native American'))
num += 1
#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'Middle Eastern'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'Middle Eastern'))
num += 1
#%%
#This cell to change the value of ethnicities_and_pins[num][1] to 'Other'
ethnicities_and_pins.append((ethnicities_and_pins_temp[num][0], 'Other'))
num += 1

#%%
#Now let's save this as a csv file

csv = pd.DataFrame(ethnicities_and_pins, columns = ['PIN ID','Identity'])
csv.to_csv(output+'participants_identities.csv')
#%%
#Let's also create a table with the direct comparison of original and final ethnicities
#We just need to take the tuple from ethnicities_and_pins_temp, put it in the first two 
#columns of a new dataframe, and add the third column with the second value from each 
#corresponding tupole in ethnicities_and_pins

final_ethnicities = pd.DataFrame(data = [{'PIN ID':ethnicities_and_pins_temp[i][0], \
            'Original Identity as Typed In':ethnicities_and_pins_temp[i][1], \
            'Final Categorization':ethnicities_and_pins[i][1]} for i in range(len(ethnicities_and_pins_temp))], columns=['PIN ID','Original Identity as Typed In','Final Categorization'])

#%%
#Let's save this as a csv file
final_ethnicities.to_csv(output+'final_ethnicities.csv')
                   
#%%
#Let's load the final_ethnicities.csv and write a print statement to retrieve the ethnicity by PIN ID
import pandas as pd
output = '/N/slate/ksalibay/Behavioral_Data/Data_Nov23/figures/'
final_ethnicities = pd.read_csv(output+'final_ethnicities.csv')
#%%
pin = '52913'

print('The ethnicity of participant with PIN ID '+pin+' was '+final_ethnicities[final_ethnicities['PIN ID'] == int(pin)]['Final Categorization'].values[0])
#Let's also print the original 
print('The original identity of participant with PIN ID '+pin+' was '+final_ethnicities[final_ethnicities['PIN ID'] == int(pin)]['Original Identity as Typed In'].values[0])

#%%
#We need to report the versions of packages that we used in this analysis
#Let's check which numpy, pandas, and UMAP versions we used

import numpy as np
import pandas as pd
import umap
import matplotlib

print('The versions of numpy, pandas, and UMAP used in this analysis were: \n numpy: '+np.__version__+
        '\n pandas: '+pd.__version__+
        '\n UMAP: '+umap.__version__+
        '\n matplotlib: '+matplotlib.__version__)