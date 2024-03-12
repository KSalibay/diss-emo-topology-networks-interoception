#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 01:04:34 2022

@author: ksalibay
This script is designed to compare the edges of two participants and find the common ones.
It follows the logic of neighboring parcels in the Destrieux atlas, and it is designed to work with the output of the .csv files
that are created by the Statistics_Differences.py script.
"""
#%%
import numpy as np
import pandas as pd

participant_label = 's'
band = 'gamma'
parts_to_compare = ['s2','S4']

sep = '-'

stats = []

for subj_num in parts_to_compare:
    fname = '/N/slate/ksalibay/DataICM2019/' + \
        str(subj_num)+'_data/'+band+'_diff_stats.csv'
    # temp = np.loadtxt(fname, delimiter=',',
    #                   dtype={'names': ('z-score', 'p-value', 'edge'),'formats': ('float', 'float', 'str')}, encoding=None)
    temp = pd.read_csv(fname, sep=',', names=['z-score', 'p-value', 'edge'])
    stats.append(temp)
# %%

# left_nodes = []
# right_nodes = []

# for i in range(len(stats)):
#     for edge in range(len(stats[i])):
#         left_nodes[edge], lh, right_nodes[edge] = str.partition(stats[i]['edge'[edge]]], 'lh')
#         left_nodes[edge], rh, right_nodes[edge] = str.partition(stats[i][2])

edges = []
number_of_participants = 2
for subj_num in range(number_of_participants):
    temp = stats[subj_num]
    edges.append(temp['edge'])

# %%
#Let's create separate entries for left and right nodes. This is going to be messy. We will clean it up in the next cell.

nodes = [[] for _ in range(number_of_participants)]

for part in range(len(edges)):
    temp1 = edges[part]
    for edge in range(len(temp1)):
        nodes[part].append((temp1[edge].split('-TO-')))
        
#%%
#Clean it up! This is an imperfect method since it might require a rerun... but just rerunning the cell would generally take care of stuff.

def contains(tuple, given_char):
    for ch in tuple:
        if ch == given_char:
            return True
    return False

for num in nodes:
    for each in num:
        if contains(each, ''):
           num.remove(each) 
          
#%%
#Final clean version of the lists of edges.

nodes_clean = [[] for _ in range(len(nodes))]
#nodes_clean = nodes.copy()
for part in range(len(nodes)):
    temp2 = nodes[part]
    for row in range(len(temp2)):
        #tuple_temp = (temp2[row][0]+temp2[row][1],temp2[row][2][1:])
        tuple_temp = (*temp2[row],)
        nodes_clean[part].append(tuple_temp)

#%%

def compare(tuple1, tuple2):
    trues = []
    for i in tuple1:
        if i in tuple2:
            trues.append(True)
    if True in trues:
        return True
    else:
        return False

commons = []

for part in range(len(nodes_clean)-1):
    temp3 = nodes_clean[part]
    temp4 = nodes_clean[part+1]
    for each_tuple in range(len(temp3)):
        for each_tuple2 in range(len(temp4)):
            if compare(temp3[each_tuple], temp4[each_tuple2]):
                commons.append((temp3[each_tuple],temp4[each_tuple2]))
                
#%%
#Then we view _commons_ as we please (in Spyder, I prefer to use Variable Explorer and just view it interactively)
#The nodes we will delete can be documented in a commented-out line here or listed in a list to pass down to .pop()

#alpha: there were 402 corrs identified; deleted: 0, 1, 2, 3, 4, 5, 7, 8, 9, 


#Let's try to optimize this code for comparisons across neighboring regions 
#(since the Destrieux figure shows numbered areas, the comparisons will go faster if we just recode the names...)

labels_indices = pd.read_csv("/N/slate/ksalibay/DataICM2019/destrieux_parcels_indices.csv", sep=',', names=['number','label'])

commons_recoded = []

for row in range(len(commons)):
    across_parts = []
    for tuple in range(len(commons[row])):
        ind0 = labels_indices[labels_indices['label']==commons[row][tuple][0][:-3]].index.values
        ind1 = labels_indices[labels_indices['label']==commons[row][tuple][1][:-3]].index.values
        if commons[row][tuple][0][-2:] == 'rh':
            it0 = str(ind0 + 1) + 'R'
        else:
            it0 = str(ind0 + 1) + 'L'
        if commons[row][tuple][1][-2:] == 'rh':
            it1 = str(ind1 + 1) + 'R'
        else:
            it1 = str(ind1 + 1) + 'L'
        across_parts.append((it0,it1))
    commons_recoded.append(across_parts)    

#%%
#Let's check that commons_recoded doesn't have any cross-hemispheric differences 
#(meaning if participant 1 has R-L that part 2 has L-L or something)

for item in range(len(commons_recoded)):
    part11 = commons_recoded[item][0][0][-1:]
    part12 = commons_recoded[item][0][1][-1:]
    part21 = commons_recoded[item][1][0][-1:]
    part22 = commons_recoded[item][1][1][-1:]
    if part11 == part12:
        if part21 != part22 or part21 != part11:
            commons_recoded.pop(item)
    elif part21 == part22:
        if part11 != part12 or part11 != part21:
            commons_recoded.pop(item)
            
#%%
#Let's remove those regions that are not neighboring each other on the Destrieux atlas.
#For that, we will write a dictionary that will reference lists for each one of the regions that are neighbors.
#Then we will loop through the commons_recoded list and remove those that are not neighbors.
            
#Let's start by creating a dictionary with the indices in the form '[number]L/R',
#so that we can easily compare them to the indices in commons_recoded.
neighbors = {'[1]L':['[5]L','[24]L','[64]L','[31]L'],
            '[2]L':['[59]L','[19]L','[37]L','[21]L','[51]L','[42]L'],
            '[3]L':['[69]L','[16]L','[46]L','[27]L','[3]R','[29]L','[45]L','[28]L','[67]L'],
            '[4]L':['[12]L','[29]L','[45]L','[28]L','[67]L','[26]L','[41]L','[49]L'],
            '[5]L':['[1]L','[24]L','[62]L','[15]L','[53]L','[16]L','[31]L'],
            '[6]L':['[6]R','[32]L','[16]L','[7]L','[66]L','[70]L'],
            '[7]L':['[7]R','[16]L','[8]L','[66]L'],
            '[8]L':['[8]R','[7]L','[16]L','[9]L','[66]L','[46]L','[3]L'],
            '[9]L':['[9]R','[8]L','[66]L','[10]L','[30]L','[71]L','[46]L'],
            '[10]L':['[10]R','[9]L','[30]L','[65]L','[44]L','[22]L','[23]L','[66]L'],
            '[11]L':['[11]R','[20]L','[65]L','[44]L','[22]L','[42]L','[58]L'],
            '[12]L':['[4]L','[29]L','[68]L','[52]L','[14]L','[40]L','[49]L','[35]L'],
            '[13]L':['[14]L','[39]L','[24]L','[62]L'],
            '[14]L':['[13]L','[62]L','[15]L','[52]L','[12]L','[40]L','[47]L','[39]L'],
            '[15]L':['[14]L','[62]L','[24]L','[5]L','[53]L','[16]L','[54]L','[52]L','[29]L','[68]L'],
            '[16]L':['[15]L','[54]L','[69]L','[29]L','[3]L','[16]R','[5]L','[70]L','[31]L','[32]L','[6]L',
                     '[7]L','[8]L'],
            '[17]L':['[18]L','[49]L','[41]L','[48]L'],
            '[18]L':['[17]L','[49]L','[47]L','[24]L','[48]L'],
            '[19]L':['[25]L','[2]R','[59]L','[57]L','[58]L','[42]L'],
            '[20]L':['[11]L','[65]L','[27]L','[56]L','[58]L','[20]R'],
            '[21]L':['[2]L','[51]L','[61]L','[50]R','[51]L','[60]L'],
            '[22]L':['[61]L','[51]L','[42]L','[11]L','[44]L','[23]L','[10]L'],
            '[23]L':['[22]L','[44]L','[10]L','[66]L','[35]L','[43]L','[50]L','[61]L'],
            '[24]L':['[1]L','[5]L','[13]L','[62]L','[15]L','[47]L','[39]L','[64]L','[63]L','[31]L'],
            '[25]L':['[19]L','[59]L','[38]L','[73]L','[55]L','[26]L','[56]L','[58]L'],
            '[26]L':['[25]L','[55]L','[56]L','[67]L','[28]L','[4]L','[41]L','[36]L','[34]L','[73]L'],
            '[27]L':['[20]L','[65]L','[30]L','[3]L','[56]L','[46]L','[67]L','[27]R'],
            '[28]L':['[26]L','[67]L','[4]L','[45]L','[3]L'],
            '[29]L':['[15]L','[68]L','[12]L','[4]L','[45]L','[3]L','[16]L','[69]L','[54]L'],
            '[30]L':['[27]L','[9]L','[65]L','[10]L','[71]L','[46]L','[30]R'],
            '[31]L':['[16]L','[5]L','[32]L','[70]L','[1]L','[24]L','[63]L','[31]R'],
            '[32]L':['[31]L','[63]L','[70]L','[6]L','[15]L','[32]R'],
            '[33]L':['[34]L','[74]L','[41]L','[48]L','[35]L'],
            '[34]L':['[33]L','[74]L','[36]L','[26]L','[73]L','[38]L','[43]L','[35]L'],
            '[35]L':['[34]L','[43]L','[23]L','[48]L','[33]L'],
            '[36]L':['[34]L','[26]L','[41]L','[74]L','[33]L'],
            '[37]L':['[2]L','[72]L','[38]L','[43]L','[50]L','[21]L','[60]L','[59]L'],
            '[38]L':['[37]L','[43]L','[34]L','[73]L','[25]L','[59]L','[72]L'],
            '[39]L':['[13]L','[14]L','[47]L','[24]L','[62]L'],
            '[40]L':['[14]L','[12]L','[49]L','[47]L'],
            '[41]L':['[36]L','[26]L','[4]L','[49]L','[17]L','[48]L','[33]L','[74]L'],
            '[42]L':['[2]L','[58]L','[11]L','[44]L','[22]L','[51]L'],
            '[43]L':['[35]L','[34]L','[38]L','[72]L','[37]L','[23]L','[50]L'],
            '[44]L':['[44]R','[42]L','[22]L','[23]L','[10]L','[65]L','[51]L'],
            '[45]L':['[28]L','[4]L','[29]L','[3]L'],
            '[46]L':['[27]L','[3]L','[8]L','[9]L','[30]L','[71]L'],
            '[47]L':['[18]L','[24]L','[13]L','[14]L','[39]L','[49]L'],
            '[48]L':['[17]L','[49]L','[41]L','[33]L','[35]L','[18]L'],
            '[49]L':['[17]L','[41]L','[4]L','[48]L','[47]L','[40]L','[12]L'],
            '[50]L':['[43]L','[23]L','[61]L','[21]L','[37]L'],
            '[51]L':['[21]L','[42]L','[22]L','[2]L','[61]L'],
            '[52]L':['[15]L','[14]L','[12]L','[68]L'],
            '[53]L':['[15]L','[5]L','[54]L'],
            '[54]L':['[53]L','[15]L','[16]L','[69]L','[29]L'],
            '[55]L':['[25]L','[26]L','[73]L'],
            '[56]L':['[20]L','[58]L','[25]L','[26]L','[27]L','[67]L'],
            '[57]L':['[19]L','[42]L','[58]L'],
            '[58]L':['[57]L','[19]L','[20]L','[56]L','[25]L','[42]L','[11]L'],
            '[59]L':['[19]L','[2]L','[37]L','[38]L','[72]L','[73]L','[25]L'],
            '[60]L':['[21]L','[37]L','[72]L'],
            '[61]L':['[21]L','[50]L','[23]L','[22]L','[51]L'],
            '[62]L':['[24]L','[13]L','[14]L','[15]L'],
            '[63]L':['[31]L','[32]L','[24]L'],
            '[64]L':['[1]L','[24]L'],
            '[65]L':['[20]L','[27]L','[30]L','[10]L','[44]L','[11]L'],
            '[66]L':['[66]R','[32]L','[6]L','[7]L','[8]L','[9]L','[10]L','[23]L'],
            '[67]L':['[26]L','[56]L','[27]L','[28]L','[4]L','[3]L'],
            '[68]L':['[15]L','[29]L','[12]L','[52]L'],
            '[69]L':['[16]L','[54]L','[29]L'],
            '[70]L':['[70]R','[16]L','[32]L','[31]L'],
            '[71]L':['[71]R','[30]L','[9]L','[46]L'],
            '[72]L':['[37]L','[38]L','[59]L','[43]L','[60]L'],
            '[73]L':['[25]L','[26]L','[38]L','[59]L','[34]L','[55]L'],
            '[74]L':['[33]L','[34]L','[36]L','[41]L'],
            '[1]R':['[5]R','[24]R','[64]R','[31]R'],
            '[2]R':['[59]R','[19]R','[37]R','[21]R','[51]R','[42]R'],
            '[3]R':['[69]R','[16]R','[46]R','[27]R','[3]L','[29]R','[45]R','[28]R','[67]R'],
            '[4]R':['[12]R','[29]R','[45]R','[28]R','[67]R','[26]R','[41]R','[49]R'],
            '[5]R':['[1]R','[24]R','[62]R','[15]R','[53]R','[16]R','[31]R'],
            '[6]R':['[6]L','[32]R','[16]R','[7]R','[66]R','[70]R'],
            '[7]R':['[7]L','[16]R','[8]R','[66]R'],
            '[8]R':['[8]L','[7]R','[16]R','[9]R','[66]R','[46]R','[3]R'],
            '[9]R':['[9]L','[8]R','[66]R','[10]R','[30]R','[71]R','[46]R'],
            '[10]R':['[10]L','[9]R','[30]R','[65]R','[44]R','[22]R','[23]R','[66]R'],
            '[11]R':['[11]L','[20]R','[65]R','[44]R','[22]R','[42]R','[58]R'],
            '[12]R':['[4]R','[29]R','[68]R','[52]R','[14]R','[40]R','[49]R','[35]R'],
            '[13]R':['[14]R','[39]R','[24]R','[62]R'],
            '[14]R':['[13]R','[62]R','[15]R','[52]R','[12]R','[40]R','[47]R','[39]R'],
            '[15]R':['[14]R','[62]R','[24]R','[5]R','[53]R','[16]R','[54]R','[52]R','[29]R','[68]R'],
            '[16]R':['[15]R','[54]R','[69]R','[29]R','[3]R','[16]L','[5]R','[70]R','[31]R','[32]R','[6]R',
                      '[7]R','[8]R'],
            '[17]R':['[18]R','[49]R','[41]R','[48]R'],
            '[18]R':['[17]R','[49]R','[47]R','[24]R','[48]R'],
            '[19]R':['[25]R','[2]L','[59]R','[57]R','[58]R','[42]R'],
            '[20]R':['[11]R','[65]R','[27]R','[56]R','[58]R','[20]L'],
            '[21]R':['[2]R','[51]R','[61]R','[50]L','[51]R','[60]R'],
            '[22]R':['[61]R','[51]R','[42]R','[11]R','[44]R','[23]R','[10]R'],
            '[23]R':['[22]R','[44]R','[10]R','[66]R','[35]R','[43]R','[50]R','[61]R'],
            '[24]R':['[1]R','[5]R','[13]R','[62]R','[15]R','[47]R','[39]R','[64]R','[63]R','[31]R'],
            '[25]R':['[19]R','[59]R','[38]R','[73]R','[55]R','[26]R','[56]R','[58]R'],
            '[26]R':['[25]R','[55]R','[56]R','[67]R','[28]R','[4]R','[41]R','[36]R','[34]R','[73]R'],
            '[27]R':['[20]R','[65]R','[30]R','[3]R','[56]R','[46]R','[67]R','[27]L'],
            '[28]R':['[26]R','[67]R','[4]R','[45]R','[3]R'],
            '[29]R':['[15]R','[68]R','[12]R','[4]R','[45]R','[3]R','[16]R','[69]R','[54]R'],
            '[30]R':['[27]R','[9]R','[65]R','[10]R','[71]R','[46]R','[30]L'],
            '[31]R':['[16]R','[5]R','[32]R','[70]R','[1]R','[24]R','[63]R','[31]L'],
            '[32]R':['[31]R','[63]R','[70]R','[6]R','[15]R','[32]L'],
            '[33]R':['[34]R','[74]R','[41]R','[48]R','[35]R'],
            '[34]R':['[33]R','[74]R','[36]R','[26]R','[73]R','[38]R','[43]R','[35]R'],
            '[35]R':['[34]R','[43]R','[23]R','[48]R','[33]R'],
            '[36]R':['[34]R','[26]R','[41]R','[74]R','[33]R'],
            '[37]R':['[2]R','[72]R','[38]R','[43]R','[50]R','[21]R','[60]R','[59]R'],
            '[38]R':['[37]R','[43]R','[34]R','[73]R','[25]R','[59]R','[72]R'],
            '[39]R':['[13]R','[14]R','[47]R','[24]R','[62]R'],
            '[40]R':['[14]R','[12]R','[49]R','[47]R'],
            '[41]R':['[36]R','[26]R','[4]R','[49]R','[17]R','[48]R','[33]R','[74]R'],
            '[42]R':['[2]R','[58]R','[11]R','[44]R','[22]R','[51]R'],
            '[43]R':['[35]R','[34]R','[38]R','[72]R','[37]R','[23]R','[50]R'],
            '[44]R':['[44]L','[42]R','[22]R','[23]R','[10]R','[65]R','[51]R'],
            '[45]R':['[28]R','[4]R','[29]R','[3]R'],
            '[46]R':['[27]R','[3]R','[8]R','[9]R','[30]R','[71]R'],
            '[47]R':['[18]R','[24]R','[13]R','[14]R','[39]R','[49]R'],
            '[48]R':['[17]R','[49]R','[41]R','[33]R','[35]R','[18]R'],
            '[49]R':['[17]R','[41]R','[4]R','[48]R','[47]R','[40]R','[12]R'],
            '[50]R':['[43]R','[23]R','[61]R','[21]R','[37]R'],
            '[51]R':['[21]R','[42]R','[22]R','[2]R','[61]R'],
            '[52]R':['[15]R','[14]R','[12]R','[68]R'],
            '[53]R':['[15]R','[5]R','[54]R'],
            '[54]R':['[53]R','[15]R','[16]R','[69]R','[29]R'],
            '[55]R':['[25]R','[26]R','[73]R'],
            '[56]R':['[20]R','[58]R','[25]R','[26]R','[27]R','[67]R'],
            '[57]R':['[19]R','[42]R','[58]R'],
            '[58]R':['[57]R','[19]R','[20]R','[56]R','[25]R','[42]R','[11]R'],
            '[59]R':['[19]R','[2]R','[37]R','[38]R','[72]R','[73]R','[25]R'],
            '[60]R':['[21]R','[37]R','[72]R'],
            '[61]R':['[21]R','[50]R','[23]R','[22]R','[51]R'],
            '[62]R':['[24]R','[13]R','[14]R','[15]R'],
            '[63]R':['[31]R','[32]R','[24]R'],
            '[64]R':['[1]R','[24]R'],
            '[65]R':['[20]R','[27]R','[30]R','[10]R','[44]R','[11]R'],
            '[66]R':['[66]L','[32]R','[6]R','[7]R','[8]R','[9]R','[10]R','[23]R'],
            '[67]R':['[26]R','[56]R','[27]R','[28]R','[4]R','[3]R'],
            '[68]R':['[15]R','[29]R','[12]R','[52]R'],
            '[69]R':['[16]R','[54]R','[29]R'],
            '[70]R':['[70]L','[16]R','[32]R','[31]R'],
            '[71]R':['[71]L','[30]R','[9]R','[46]R'],
            '[72]R':['[37]R','[38]R','[59]R','[43]R','[60]R'],
            '[73]R':['[25]R','[26]R','[38]R','[59]R','[34]R','[55]R'],
            '[74]R':['[33]R','[34]R','[36]R','[41]R']}
#%%
#Now let's loop through the commons_recoded list. Each row is a pair of two-item lists, in which one item is common in both lists, and the other is not.
#Let's check if the uncommon items are neighbors. If they are not, we will remove the row from the list.
#Let's also keep the rows that are the same in both lists.
#The order of common and uncommon items is not important, so we will check both ways.

commons_recoded_final = []

for row in range(len(commons_recoded)):
    part11 = commons_recoded[row][0][0]
    part12 = commons_recoded[row][0][1]
    part21 = commons_recoded[row][1][0]
    part22 = commons_recoded[row][1][1]
    if part11 == part21:
        if part12 == part22:
            commons_recoded_final.append(commons_recoded[row])
        else:
            if part12 in neighbors[part22]:
                commons_recoded_final.append(commons_recoded[row])
    else:
        if part11 in neighbors[part21]:
            if part12 == part22:
                commons_recoded_final.append(commons_recoded[row])
            else:
                if part12 in neighbors[part22]:
                    commons_recoded_final.append(commons_recoded[row])
        else:
            if part12 in neighbors[part21]:
                if part11 == part22:
                    commons_recoded_final.append(commons_recoded[row])
                else:
                    if part11 in neighbors[part22]:
                        commons_recoded_final.append(commons_recoded[row])

commons_recoded = commons_recoded_final
#%%
#Now let's decode the indices back into words...

commons_decoded = []

for row in range(len(commons_recoded)):
    both = []
    for tuple in range(len(commons_recoded[row])):
        parc11 = labels_indices['label'][int(commons_recoded[row][tuple][0][1:-2])-1]
        parc12 = labels_indices['label'][int(commons_recoded[row][tuple][1][1:-2])-1]
        parc11 += "-" + commons_recoded[row][tuple][0][-1:]
        parc12 += "-" + commons_recoded[row][tuple][1][-1:]
        both.append((parc11,parc12))
    commons_decoded.append(both)

#And let's save it to a .csv
commons_decoded_final = []
for row in commons_decoded:
    temp1 = row[0][0] + '-TO-' + row[0][1]
    temp2 = row[1][0] + '-TO-' + row[1][1]
    
    commons_decoded_final.append((temp1,temp2))
    
fname = '/N/slate/ksalibay/DataICM2019/'+parts_to_compare[1]+'_data/' + band + '_' \
+ parts_to_compare[0] + 'to' + parts_to_compare[1] + '_common_edges_new.csv'

import csv
with open(fname,'wt') as out:
   csv_out=csv.writer(out)
   csv_out.writerows(commons_decoded_final)
#%%
#This cell exists purely to pull out z-scores and p-values corresponding to the edges we identified earlier
#We will also create a large Pandas Dataframe with all of the data combined in the next step
stats_temp_z_1 = []
stats_temp_p_1 = []
stats_temp_z_2 = []
stats_temp_p_2 = []
part1_edges = []
part2_edges = []

for row in commons_decoded:
    newrow = []
    for edge in row:
        newedge = []
        for node in edge:
            if node[-1:] == 'R':
                node1 = node[:-1] + 'rh'
            else:
                node1 = node[:-1] + 'lh'
            newedge.append(node1)
        newrow.append(newedge)
    part1_edges.append(newrow[0])
    part2_edges.append(newrow[1])
    
part1_data = stats[0]
part2_data = stats[1]

for each in range(len(part1_edges)):
    str_temp = part1_edges[each][0] + sep + part1_edges[each][1]
    idx = part1_data[part1_data['edge'] == str_temp].index.tolist()
    stats_temp_z_1.append(part1_data['z-score'][0])
    stats_temp_p_1.append(part1_data['p-value'][0])
    
for each in range(len(part2_edges)):
    str_temp = part2_edges[each][0] + sep + part2_edges[each][1]
    idx = part2_data[part2_data['edge'] == str_temp].index.tolist()
    stats_temp_z_2.append(part2_data['z-score'][0])
    stats_temp_p_2.append(part2_data['p-value'][0])

#%%
#Let's also figure out which overarching lobar structures do our edges project to and from...

import pickle
with open("/N/slate/ksalibay/DataICM2019/aparc2009s_labels.txt", "rb") as fp:
    labels = pickle.load(fp)
labels = labels[:148]

frontal = [6,7,12,13,14,15,24,25,26,27,28,29,30,31,32,33,46,47,56,57,60,61,88,89,
           102,103,104,105,106,107,122,123,124,125,126,127,134,135,136,137,138,139]
ins = [16,17,34,35,76,77,78,79,80,81,92,93,94,95,96,97]
lim = [0,1,2,3,4,5,18,19,20,21,62,63,90,91,130,131]
tem = [64,65,66,67,68,69,70,71,72,73,74,75,84,85,98,99,142,143,144,145,146,147]
par = [10,11,48,49,50,51,52,53,54,55,58,59,108,109,110,111,128,129,132,133,140,141]
occ = [8,9,22,23,36,37,38,39,40,41,42,43,44,45,82,83,86,87,100,101,112,113,114,115,
       116,117,118,119,120,121]
lobes_1 = [None]* len(part1_edges)
lobes_2 = [None]* len(part1_edges)
lobes_3 = [None]* len(part2_edges)
lobes_4 = [None]* len(part2_edges)

for i in range(len(part1_edges)):
    node1 = part1_edges[i][0]
    id_node1 = labels.index(node1)
    if id_node1 in frontal:
        lobes_1[i] = 'Frontal'
    elif id_node1 in ins:
        lobes_1[i] = 'Insular'
    elif id_node1 in lim:
        lobes_1[i] = 'Limbic'
    elif id_node1 in tem:
        lobes_1[i] =  'Temporal'
    elif id_node1 in par:
        lobes_1[i] =  'Parietal'
    elif id_node1 in occ:
        lobes_1[i] = 'Occipital'
        
for j in range(len(part1_edges)):
    node2 = part1_edges[j][1]
    id_node2 = labels.index(node2)
    if id_node2 in frontal:
        lobes_2[j] = 'Frontal'
    elif id_node2 in ins:
        lobes_2[j] = 'Insular'
    elif id_node2 in lim:
        lobes_2[j] = 'Limbic'
    elif id_node2 in tem:
        lobes_2[j] =  'Temporal'
    elif id_node2 in par:
        lobes_2[j] =  'Parietal'
    elif id_node2 in occ:
        lobes_2[j] = 'Occipital'
        
for k in range(len(part2_edges)):
    node3 = part2_edges[k][0]
    id_node3 = labels.index(node3)
    if id_node3 in frontal:
        lobes_3[k] = 'Frontal'
    elif id_node3 in ins:
        lobes_3[k] = 'Insular'
    elif id_node3 in lim:
        lobes_3[k] = 'Limbic'
    elif id_node3 in tem:
        lobes_3[k] =  'Temporal'
    elif id_node3 in par:
        lobes_3[k] =  'Parietal'
    elif id_node3 in occ:
        lobes_3[k] = 'Occipital'
        
for l in range(len(part2_edges)):
    node4 = part2_edges[l][1]
    id_node4 = labels.index(node4)
    if id_node4 in frontal:
        lobes_4[l] = 'Frontal'
    elif id_node4 in ins:
        lobes_4[l] = 'Insular'
    elif id_node4 in lim:
        lobes_4[l] = 'Limbic'
    elif id_node4 in tem:
        lobes_4[l] =  'Temporal'
    elif id_node4 in par:
        lobes_4[l] =  'Parietal'
    elif id_node4 in occ:
        lobes_4[l] = 'Occipital'
    
#%%
#Here is the mother-of-all-dataframes.
dest_nums_1 = [item[0] for item in commons_recoded]
dest_nums_2 = [item[1] for item in commons_recoded]
dest_labels_1 = [row[0] for row in commons_decoded_final]
dest_labels_2 = [row[1] for row in commons_decoded_final]

data = [dest_nums_1, dest_nums_2, dest_labels_1, dest_labels_2, stats_temp_z_1, stats_temp_p_1, stats_temp_z_2, stats_temp_p_2,
        lobes_1, lobes_2, lobes_3, lobes_4]

full_dataframe = pd.DataFrame(data)
full_dataframe = full_dataframe.transpose()
full_dataframe.columns = ['Destrieux_nums_P1', 'Destrieux_nums_P2', 'Destrieux_labels_P1', 'Destrieux_labels_P2', 
                          'Z-score_P1', 'P-value_P1', 'Z-score_P2', 'P-value_P2', 'Lobe_left_P1', 'Lobe_right_P1',
                          'Lobe_left_P2', 'Lobe_right_P2']

fname2 = '/N/slate/ksalibay/DataICM2019/'+parts_to_compare[1] + '_data/' + band + '_' \
+ parts_to_compare[0] + 'to' + parts_to_compare[1] + '_common_edges_full_new.csv'
full_dataframe.to_csv(fname2)
# %%
