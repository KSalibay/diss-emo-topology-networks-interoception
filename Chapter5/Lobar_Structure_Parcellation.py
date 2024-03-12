#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:21:32 2022

@author: ksalibay
"""
import scipy.io
import numpy as np

#Define your canonical frequency band of interest (presumes you have already run the community detection algorithm)
#Also specify your subject and the folder where the data is located
band = 'gamma'
subject = 's5'
folder = '/N/slate/ksalibay/DataICM2019/'+subject+'_data/'

diastole1 = scipy.io.loadmat(file_name = folder+band+'_diastole/MATs/'+subject+'nounkdiastole_'+band+'_comms')
systole1 = scipy.io.loadmat(file_name = folder+band+'_systole/MATs/'+subject+'nounksystole_'+band+'_comms')

##%%

diastole2 = diastole1['ciu1'].squeeze()
systole2 = systole1['ciu2'].squeeze()

##%%
#Load the labels for aparc2009s parcellation in FreeSurfer
import pickle
with open("/N/slate/ksalibay/DataICM2019/aparc2009s_labels.txt", "rb") as fp:
    labels = pickle.load(fp)
labels = labels[:148]
labels = np.asarray(labels)
    
##%%
#Create arrays for diastole and systole with the labels
diastole_fin = np.column_stack((diastole2,labels))
systole_fin = np.column_stack((systole2,labels))

##%%

# diastole_fin2 = diastole_fin[diastole_fin[:,0].argsort()]
# systole_fin2 = systole_fin[systole_fin[:,0].argsort()]

##%%

#np.savetxt('/N/slate/ksalibay/DataICM2019/'+subject+'_conmats/'+subject+'_diastole_'+band+'.txt',diastole_fin,delimiter=',',fmt='%s')
#np.savetxt('/N/slate/ksalibay/DataICM2019/'+subject+'_conmats/'+subject+'_systole_'+band+'.txt',systole_fin,delimiter=',',fmt='%s')

##%%

#Let's count how large each community is

i_dia = max(diastole_fin[:,0].astype(int))
unique_dia, counts_dia = np.unique(diastole_fin[:,0],return_counts=True)
comm_sort_dia = np.asarray((unique_dia, counts_dia)).T
dia_count_ind = np.argsort(-counts_dia)
dia_sort_ind = unique_dia[dia_count_ind]

i_sys = max(systole_fin[:,0].astype(int))
unique_sys, counts_sys = np.unique(systole_fin[:,0],return_counts=True)
comm_sort_sys = np.asarray((unique_sys, counts_sys)).T
sys_count_ind = np.argsort(-counts_sys)
sys_sort_ind = unique_sys[sys_count_ind]

#%%
#Resorting the communities and renumbering them in accordance with the largest to smallest order
dia_fin = diastole_fin.copy()
for i in range(len(dia_sort_ind)):
    temp_list = np.where(dia_sort_ind[i]==diastole_fin)
    temp_list = temp_list[0]
    for j in temp_list:
        dia_fin[j,0] = i+1
        dia_fin[j,1] = diastole_fin[j,1]
        
sys_fin = systole_fin.copy()
for k in range(len(sys_sort_ind)):
    temp_list = np.where(sys_sort_ind[k]==systole_fin)
    temp_list = temp_list[0]
    for l in temp_list:
        sys_fin[l,0] = k+1
        sys_fin[l,1] = systole_fin[l,1]
#%%
#Let's create the combined diastole-systole correspondence data frame
sankey_full = dia_fin.copy()
sankey_full = np.insert(sankey_full,2,sys_fin[:,0],axis=1)
sankey_full = sankey_full[:,[1,0,2]]

#%%

#Let's also sort out the labels in df into corresponding lobar structure 
#— taken from Irimia, A., Chambers, M. C., Torgerson, C. M., & Van Horn, J. D. (2012). 
#Circular representation of human cortical networks for subject and population-level connectomic visualization. Neuroimage, 60(2), 1340-1351.
#Save the data here with the lobe labels
frontal = [6,7,12,13,14,15,24,25,26,27,28,29,30,31,32,33,46,47,56,57,60,61,88,89,
           102,103,104,105,106,107,122,123,124,125,126,127,134,135,136,137,138,139]
ins = [16,17,34,35,76,77,78,79,80,81,92,93,94,95,96,97]
lim = [0,1,2,3,4,5,18,19,20,21,62,63,90,91,130,131]
tem = [64,65,66,67,68,69,70,71,72,73,74,75,84,85,98,99,142,143,144,145,146,147]
par = [10,11,48,49,50,51,52,53,54,55,58,59,108,109,110,111,128,129,132,133,140,141]
occ = [8,9,22,23,36,37,38,39,40,41,42,43,44,45,82,83,86,87,100,101,112,113,114,115,
       116,117,118,119,120,121]
sankey_full1 = np.zeros(len(labels))
sankey_full1 = sankey_full1.astype(str)
for i in range(len(labels)):
    if (sankey_full[i,0] == diastole_fin[i,1]) and (i in frontal):
        sankey_full1[i] = 'Frontal'
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in ins):
        sankey_full1[i] = 'Insular'
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in lim):
        sankey_full1[i] = 'Limbic'
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in tem):
        sankey_full1[i] = 'Temporal'
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in par):
        sankey_full1[i] = 'Parietal'
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in occ):
        sankey_full1[i] = 'Occipital'

sankey_full3 = np.insert(sankey_full,3,sankey_full1,axis=1)
np.savetxt('/N/slate/ksalibay/DataICM2019/'+subject+'_conmats/'+subject+'_diastole_'+band+'_sankey_fin.txt',sankey_full3,delimiter=',',fmt='%s')

#%%
#Save the data with numbers instead of labels of lobes
frontal = [6,7,12,13,14,15,24,25,26,27,28,29,30,31,32,33,46,47,56,57,60,61,88,89,
           102,103,104,105,106,107,122,123,124,125,126,127,134,135,136,137,138,139]
ins = [16,17,34,35,76,77,78,79,80,81,92,93,94,95,96,97]
lim = [0,1,2,3,4,5,18,19,20,21,62,63,90,91,130,131]
tem = [64,65,66,67,68,69,70,71,72,73,74,75,84,85,98,99,142,143,144,145,146,147]
par = [10,11,48,49,50,51,52,53,54,55,58,59,108,109,110,111,128,129,132,133,140,141]
occ = [8,9,22,23,36,37,38,39,40,41,42,43,44,45,82,83,86,87,100,101,112,113,114,115,
       116,117,118,119,120,121]
sankey_full1 = np.zeros(len(labels))
sankey_full1 = sankey_full1.astype(int)
for i in range(len(labels)):
    if (sankey_full[i,0] == diastole_fin[i,1]) and (i in frontal):
        sankey_full1[i] = 1
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in ins):
        sankey_full1[i] = 2
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in lim):
        sankey_full1[i] = 3
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in tem):
        sankey_full1[i] = 4
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in par):
        sankey_full1[i] = 5
    elif (sankey_full[i,0] == diastole_fin[i,1]) and (i in occ):
        sankey_full1[i] = 6

sankey_full2 = np.insert(sankey_full,3,sankey_full1,axis=1)
np.savetxt('/N/slate/ksalibay/DataICM2019/'+subject+'_conmats/'+subject+'_diastole_'+band+'_sankey_fin_num.txt',sankey_full2,delimiter=',',fmt='%s')
