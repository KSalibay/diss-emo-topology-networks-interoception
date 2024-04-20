#%%
#In this script, we will try and see if we can apply some geomstats analyses to the dynamic data
#We will try to get the shape of the manifold of the dynamic data by using curve analyses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input = '/N/slate/ksalibay/Behavioral_Data/Data_Nov23/'
fname = 'pins_final.csv'
pins = pd.read_csv(input+fname)
#Let's only get column 2 from pins and convert it into a list
pins = pins.iloc[:,1]
pins = pins.tolist()
#%%
#Let's get a pin from pins 

pin = str(pins[3])
#%%
#==============================================================================
#Let's now get the indices of the files
import os
import re

#Let's first get the list of files in the directory that have the mask input+'participant_'+pin+'_row_'+str(index)+'.csv'
files = os.listdir(input)
files_to_check = []
for file in files:
    if re.match('participant_'+pin+'_row_', file):
        #Let's check if the file is a csv
        if file.endswith('.csv'):
            files_to_check.append(file)
indices = []
for file in files_to_check:
    indices.append(int(file.split('_')[-1].split('.')[0]))

#==============================================================================

#Index 0 is faulty, so let's remove it
indices = sorted(indices)
indices.remove(0)

#Let's for now pick just one index
index = indices[2]

#Now we will get the dynamic data and put it into a dataframe together with a column that will indicate the time scale in seconds

fname_dyn = input+'participant_'+pin+'_row_'+str(index)+'.csv'

#Here we will unpickle the .csv into numpy array
dyn = np.genfromtxt(fname_dyn, delimiter=',')

dyn = dyn[1:]

timescale = np.linspace(0,len(dyn),len(dyn))

#Let's form full_data by concatenating dyn and timescale
full_data = np.concatenate((dyn, timescale.reshape(-1,1)), axis=1)

#Let's check for nans in the data column-wise and stretch the columns where there are nans
#to match the length of the longest column without nans

#Let's first check for nans in the data
nans = np.isnan(full_data)

#Let's now check for columns that have nans
nans_per_column = np.sum(nans, axis=0)
cols_to_stretch = np.where(nans_per_column > 0)[0]

#Let's now stretch the columns that have nans by interpolating that column to match the length of the longest column without nans
for i in cols_to_stretch:
    col = full_data[:,i]
    col = col[~np.isnan(col)]
    col_interp = np.interp(np.linspace(0,1,len(full_data)), np.linspace(0,1,len(col)), col)
    full_data[:,i] = col_interp

#Finally, let's check that there are no more nans in the data

nans_check = np.isnan(full_data)
if nans_check.any() == False:
    print("There are no more nans in the data.")

#Let's check for redundancy in the full_data array by checking for correlation between columns

corr_matrix = np.corrcoef(full_data, rowvar=False)

#Let's check for columns that have a correlation coefficient of 1
#We will do this by checking for columns that have a correlation coefficient >=0.99 with all other columns

cols_to_drop = []
for i in range(corr_matrix.shape[0]):
    if np.all(corr_matrix[i,:] >= 0.99):
        cols_to_drop.append(i)

if cols_to_drop:
    print("The following columns have been identified as redundant and will be dropped: ", cols_to_drop)
else:
    print("There are no redundant columns in the data.")
    del cols_to_drop

#Let's now plot the data to see what it looks like

#Let's first plot the data in 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(full_data[:,0], full_data[:,1], full_data[:,2], c=full_data[:,3], cmap='viridis')
ax.set_xlabel('Body')
ax.set_ylabel('Mind')
ax.set_zlabel('Soc')
plt.show()

# %%
#Now let's import the other data and store them all in a list of arrays called data

data = []
for index in indices:
    fname_dyn = input+'participant_'+pin+'_row_'+str(index)+'.csv'
    dyn = np.genfromtxt(fname_dyn, delimiter=',')
    dyn = dyn[1:]
    timescale = np.linspace(0,len(dyn),len(dyn))
    full_data = np.concatenate((dyn, timescale.reshape(-1,1)), axis=1)
    nans = np.isnan(full_data)

    #Let's now check for columns that have nans
    nans_per_column = np.sum(nans, axis=0)
    cols_to_stretch = np.where(nans_per_column > 0)[0]

    #Let's now stretch the columns that have nans by interpolating that column to match the length of the longest column without nans
    for i in cols_to_stretch:
        col = full_data[:,i]
        col = col[~np.isnan(col)]
        col_interp = np.interp(np.linspace(0,1,len(full_data)), np.linspace(0,1,len(col)), col)
        full_data[:,i] = col_interp

    #Finally, let's check that there are no more nans in the data

    nans_check = np.isnan(full_data)
    if nans_check.any() == False:
        print("There are no more nans in the data.")

    data.append(full_data)

#Let's remove the shorter curves with len <= 5 from data
data = [x for x in data if len(x) > 5]

#Let's try to loop through data and first resample each curve to match the length of the shortest curve
#Let's first get the length of the shortest curve and store its index in data
lengths = [len(x) for x in data]
print("The shortest curve has length: ", min(lengths))
index = lengths.index(min(lengths))

# %%
#Let's now resample each curve to match the length of the shortest curve
resampled_data = []
for i in range(len(data)):
    if i != index:
        new_data = np.zeros((min(lengths), data[i].shape[1]))
        for j in range(data[i].shape[1]):
            new_data[:,j] = np.interp(np.linspace(0,1,min(lengths)), np.linspace(0,1,len(data[i])), data[i][:,j])
        resampled_data.append(new_data)
    else:
        resampled_data.append(data[i])

#Let's make sure that the resampled data has the same length
lengths_resampled = [len(x) for x in resampled_data]

#Let's go back to UMAP and visualizing all the curves
#Let's do a 2-component PCA on the data and then apply UMAP to the result of PCA
#Let's first concatenate all the curves into one array
from sklearn.decomposition import PCA
import umap
data_array = np.array(resampled_data)
#Let's first perform PCA on each curve, and then concatenate the result of PCA into one array
pca_results = []
for i in range(len(data_array)):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_array[i])
    pca_results.append(pca_result)
pca_results = np.array(pca_results)
pca_results = np.reshape(pca_results, (-1, pca_results.shape[-1]))

#Now let's fit UMAP with n_neighbors=5 and n_components=2 to the result of PCA
n_neighbors = 35
umap_model_7 = umap.UMAP(n_components=2, n_neighbors=n_neighbors,n_jobs=6)
umap_embeddings_5 = umap_model_7.fit_transform(pca_results)

#Let's visualize the result of the UMAP
fig12 = plt.figure()
ax12 = fig12.add_subplot(111)

ax12.scatter(umap_embeddings_5[:,0], umap_embeddings_5[:,1], cmap='viridis')
ax12.set_xlabel('UMAP_C1')
ax12.set_ylabel('UMAP_C2')

#Let's save the result of UMAP as a csv file
umap_df = pd.DataFrame(umap_embeddings_5)
umap_df.to_csv(input+'participant_'+pin+str(n_neighbors)+'_umap.csv', index=False)

# %%
#Let's also try that with n_neighbors = 50 and n_components = 3

n_neighbors_2 = 100
n_components_2 = 3

umap_model_8 = umap.UMAP(n_components=n_components_2, n_neighbors=n_neighbors_2)
umap_embeddings_6 = umap_model_8.fit_transform(pca_results)

#Let's visualize the result of the UMAP
fig13 = plt.figure()
ax13 = fig13.add_subplot(111, projection='3d')

ax13.scatter(umap_embeddings_6[:,0], umap_embeddings_6[:,1], umap_embeddings_6[:,2], cmap='viridis')
ax13.set_xlabel('UMAP_C1')
ax13.set_ylabel('UMAP_C2')
ax13.set_zlabel('UMAP_C3')

#Let's save the result of UMAP as a csv file
umap_df_2 = pd.DataFrame(umap_embeddings_6)
umap_df_2.to_csv(input+'participant_'+pin+'neighbors_'+str(n_neighbors_2)+'components'+str(n_components_2)+'_umap.csv', index=False)

#%%
