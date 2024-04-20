#Code for concatenating one participant's data from each row and then analyzing it all with UMAP

#%%
#Let's clear the variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import umap
import numba
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import spacy

##%%
##%%
input = '/N/slate/ksalibay/Behavioral_Data/Data_Nov23/'
fname = 'pins_final.csv'
pins = pd.read_csv(input+fname)
#Let's only get column 2 from pins and convert it into a list
pins = pins.iloc[:,1]
pins = pins.tolist()
##%%
#Let's get a pin from pins 
#If we know the pin that we want, let's just set pin to that pin
#pin = str(pins[5])
pin = '77367'
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
        if file.endswith('_lowres.csv'):
            files_to_check.append(file)
indices = []
for file in files_to_check:
    #Let's use regex to get the index from the file name that follows the pattern 'participant_'+pin+'_row_'+str(index)+'lowres.csv'
    indices.append(int(file.split('_')[-2].split('.')[0]))

#==============================================================================

#Index 0 is faulty, so let's remove it
indices = sorted(indices)
indices.remove(0)

#Let's for now pick just one index
index = indices[2]

#Now we will get the dynamic data and put it into a dataframe together with a column that will indicate the time scale in seconds

fname_dyn = input+'participant_'+pin+'_row_'+str(index)+'_lowres.csv'

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
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(full_data[:,0], full_data[:,1], full_data[:,2], c=full_data[:,3], cmap='viridis')
# ax.set_xlabel('Body')
# ax.set_ylabel('Mind')
# ax.set_zlabel('Soc')
# plt.show()
    
##%%
#Now let's import the other data and store them all in a list of arrays called data

data = []
for index in indices:
    fname_dyn = input+'participant_'+pin+'_row_'+str(index)+'_lowres.csv'
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

## %%
#Now what we want to do is carry through to this plot some labels.
#Those labels are stored in a csv file with 'participant_'+pin+'_emos_index.csv' as the mask.
#Each entry in that csv corresponds to the first and last index of a curve in resampled_data
#Let's first load the csv file

fname_emos = input+'participant_'+pin+'_emos_index_lowres.csv'
emos_index = pd.read_csv(fname_emos)
##%%
#We want to be able to put the labels from emos_index on the plot of the UMAP at the corresponding points
#Let's first create a list of labels that correspond to the first and last point of each row in data
first_labels = []
last_labels = []

for i in range(len(emos_index)):
    first_labels.append(emos_index.iloc[i,0])
    last_labels.append(emos_index.iloc[i,1])
##%%
#For each array in data, we want to create a new array, 
# with the same length as data, that will contain the labels in the last column
#The column will be filled with NaNs, but have labels in the rows corresponding to the first and last points of each row in data

data_with_labels = []
for i in range(len(data)):
    new_array = np.empty((len(data[i]), data[i].shape[1]+1), dtype=object)
    new_array[:] = np.nan
    new_array[:,0:-1] = data[i]
    new_array[0,-1] = first_labels[i]
    new_array[-1,-1] = last_labels[i]
    data_with_labels.append(new_array)
# #%%
#Print first 10 and last 10 rows of data_with_labels[0] to make sure that the labels are in the right place
print(data_with_labels[0][0:10,:])
print(data_with_labels[0][-10:,:])
## %%
#Let's check the lengths of individual arrays in data_with_labels

lengths = []
for i in range(len(data_with_labels)):
    lengths.append(len(data_with_labels[i]))

print(lengths)
## %%
#Now we want to concatenate all the arrays in data_with_labels into one array
#But we want to also interpolate the distances between the last point of one row and the first point of the next row
#Let's make that distance equal to an average time that would pass between the last point of one row and the first point of the next row
#We will take that time to equal 43200, as we did for the longest row in resampled_data.

#We will do that to all rows in data_with_labels except the last one, interpolating between the last point of one row and the first point of the next row

#We will then concatenate all the arrays in data_with_labels into one array

cont_data_temp = []
interp_value = 1440
for i in range(len(data_with_labels)-1):
    new_array = np.empty((len(data_with_labels[i])+interp_value, data_with_labels[i].shape[1]), dtype=object)
    new_array[:] = np.nan
    new_array[0:len(data_with_labels[i]),:] = data_with_labels[i]
    last_value = data_with_labels[i][-1,:]
    next_value = data_with_labels[i+1][0,:]
    #Let's interpolate those lengths for the first four columns, but not the one with labels
    #We will use last_value and next_value to interpolate
    for column in range(data_with_labels[i].shape[1]-1):
        temp_interp = np.interp(np.linspace(0,1,interp_value), np.linspace(0,1,2), [last_value[column], next_value[column]])
        new_array[len(data_with_labels[i]):len(data_with_labels[i])+interp_value,column] = temp_interp
    cont_data_temp.append(new_array)

#Let's also add the last array in data_with_labels to cont_data_temp
cont_data_temp.append(data_with_labels[-1])
## %%
#Let's check the lengths of the arrays in cont_data_temp
lengths_interp = []

for array_interp in range(len(cont_data_temp)):
    lengths_interp.append(len(cont_data_temp[array_interp]))

print(lengths_interp)
## %%
#Now let's concatenate all the arrays in cont_data_temp into one array and add the last array in data_with_labels to it

cont_data = np.concatenate(cont_data_temp, axis=0)
cont_data = np.concatenate((cont_data, data_with_labels[-1]), axis=0)

print(cont_data.shape)
## %%
#We now want to apply UMAP to cont_data
#However, let's first transform the last column to numeric data using PCA

emos = []
for i in cont_data:
    emos.append(i[-1])

##%%
#Let's also remove any NaNs from each element of emos
emos = [x for x in emos if str(x) != 'nan']
#Let's also replace the typo "Sacred" for "Scared" in emos
emos = [x if x != 'Sacred ' else 'Scared' for x in emos]
#Let's also remove trailing spaces from each element of emos
emos = [x.strip() for x in emos]
#Let's first tokenize the labels in emos_index
emos_index1 = [str(x) for x in emos]
##%%
# Load pre-trained Word2Vec model
nlp = spacy.load('en_core_web_md')

emotion_vectors = np.array([nlp(word).vector for word in emos_index1])

##%%
#Now what we have are vectors that correspond to the labels in emos_index1
#Each one of the vectors in emotion_vectors corresponds to a singular point in resampled_data

#Let's try this with PCA run on emotion_vectors with n_components=4

pca = PCA(n_components=2)

emo_pca = pca.fit_transform(emotion_vectors)
#%%
#Let's plot the result of PCA
plt.scatter(emo_pca[:, 0], emo_pca[:, 1], cmap='viridis', s=10)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('PCA projection of the Emotion labels')
for i in range(len(emo_pca)):
    plt.text(emo_pca[i,0], emo_pca[i,1], emos_index1[i], size=10, zorder=1, color='k')
plt.show()
#%%

# #Let's plot the result of PCA
# fig16 = plt.figure()
# #Let's make the figure size 800x800
# fig16.set_size_inches(8,8)
# ax16 = fig16.add_subplot(111, projection='3d')

# ax16.scatter(emo_pca[:, 0], emo_pca[:, 1], emo_pca[:, 2], cmap='viridis', s=10)
# #Let's also add the labels to the plot
# for i in range(len(emo_pca)):
#     ax16.text(emo_pca[i,0], emo_pca[i,1], emo_pca[i,2], emos_index1[i], size=10, zorder=1, color='k')
# ax16.set_xlabel('PCA_C1')
# ax16.set_ylabel('PCA_C2')
# ax16.set_zlabel('PCA_C3')
# plt.show()


# plt.scatter(emo_pca[:, 0], emo_pca[:, 1], cmap='viridis', s=10)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('PCA projection of the Emotion labels')
# plt.show()
# %%
#Let's create a new array that will contain the PCA result of emotion_vectors and the rest of the columns in cont_data
#First, let's interpolate the emotion_vectors to match the length of cont_data
#We will use the original indices of entries in emotion_vectors to interpolate the vectors

#Let's first get the lengths of the original emotion_vectors, meaning the lengths of cont_data column -1 between each pair of consecutive entries in emos_index

lengths_emo = []

#We have the lengths stored in lengths_interp plus the length of the last row in data_with_labels

lengths_emo = lengths_interp.copy()
lengths_emo.append(len(data_with_labels[-1]))
##%%
#Let's now interpolate the emotion_vectors to match the lengths in lengths_emo
emo_interp = []

for length in range(len(lengths_emo)):
    new_emo = np.zeros((lengths_emo[length], emo_pca.shape[1]))
    for j in range(emo_pca.shape[1]):
        new_emo[:,j] = np.interp(np.linspace(0,1,lengths_emo[length]), np.linspace(0,1,len(emo_pca)), emo_pca[:,j])
    emo_interp.append(new_emo)

print([len(x) for x in emo_interp])
##%%    
#Let's now concatenate the arrays in emo_interp into one array.

emo_interp_final = np.concatenate(emo_interp, axis=0)
#%%
## %%
#Now let's delete the last column in cont_data and hstack it with emo_interp_final

cont_data_final = cont_data[:,:-1]
cont_data_final = np.hstack((cont_data_final, emo_interp_final))

#Let's save cont_data_final as a .npy file
np.save(input+'participant_'+pin+'_cont_data_final.npy', cont_data_final)
## %%
#Now let's apply UMAP to cont_data_final
#First let's define a torus function

@numba.njit(fastmath=True)
def torus_euclidean_grad(x, y, torus_dimensions=(2*np.pi,2*np.pi)):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    distance_sqr = 0.0
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        a = abs(x[i] - y[i])
        if 2*a < torus_dimensions[i]:
            distance_sqr += a ** 2
            g[i] = (x[i] - y[i])
        else:
            distance_sqr += (torus_dimensions[i]-a) ** 2
            g[i] = (x[i] - y[i]) * (a - torus_dimensions[i]) / a
    distance = np.sqrt(distance_sqr)
    return distance, g/(1e-6 + distance)

##%%

#Let's save cont_data_final as a .csv file
cont_data_final1 = pd.DataFrame(cont_data_final)

cont_data_final1.to_csv(input+'participant_'+pin+'_cont_data_final_lowres.csv', index=False)
#%%
#Let's also write a line to load cont_data_final from the .csv file

#cont_data_final = pd.read_csv(input+'participant_'+pin+'_cont_data_final.csv')
##%%
#Let's run PCA on cont_data_final and then UMAP
#We will use the torus_euclidean_grad as the output_metric for UMAP
#Let's reduce the dimensionality of cont_data_final to 4 using PCA

pca1 = PCA(n_components=4)

cont_data_final_pca = pca1.fit_transform(cont_data_final)

##%%
interm_embedding = umap.UMAP(n_neighbors=5, n_components = 3).fit(cont_data_final1)

#%%
##%%
final_embedding = umap.UMAP(n_neighbors = 3, output_metric=torus_euclidean_grad).fit(cont_data_final1)
R = 5 # Size of the doughnut circle
r = 1 # Size of the doughnut cross-section

x = (R + r * np.cos(final_embedding.embedding_[:, 0])) * np.cos(final_embedding.embedding_[:, 1])
y = (R + r * np.cos(final_embedding.embedding_[:, 0])) * np.sin(final_embedding.embedding_[:, 1])
z = r * np.sin(final_embedding.embedding_[:, 0])

#Let's save the result of UMAP as a csv file
umap_df = pd.DataFrame(final_embedding.embedding_)

umap_df.to_csv(input+'participant_'+pin+'_umap_with_emolabels.csv', index=False)

##%%
#Let's now plot the labels in emos_index on the UMAP plot
#The indices of rows that correspond to our emotion words follow the pattern of lengths_interp
#We will use those indices to plot the labels on the UMAP plot

##%%
emo_indices = []
#We need a list of indices that will correspond to the indices of the rows in cont_data_final that correspond to the emotion words;
#We need to have the same number of indices as len(emos), so we will append a 0, and then the value of lengths_interp[0]-1,
#then value of lengths_interp[0], then value of lengths_interp[0]+lengths_interp[1]-1, then value of lengths_interp[0]+lengths_interp[1], and so on
#We also need to take care of the last pair corresponding to the last value of lengths_interp

for i in range(len(lengths_interp)):
    if i == 0:
        emo_indices.append(0)
        emo_indices.append(lengths_interp[i]-1)
    elif i == len(lengths_interp)-1:
        emo_indices.append(sum(lengths_interp[0:i]))
        emo_indices.append(sum(lengths_interp[0:i])+lengths_interp[i]-1)
    else:
        emo_indices.append(sum(lengths_interp[0:i]))
        emo_indices.append(sum(lengths_interp[0:i])+lengths_interp[i]-1)


##%%
x_labels = x[emo_indices]
y_labels = y[emo_indices]
z_labels = z[emo_indices]
##%%
#Let's plot this as a figure with a big enough size, say, 800x800
fig18 = plt.figure()
#Let's make the figure size 800x800
fig18.set_size_inches(8,8)
ax18 = fig18.add_subplot(111, projection='3d')

ax18.scatter(x, y, z, cmap='viridis', s=0.1)
#We now want to add the labels to the plot at the corresponding coordinates of x_labels, y_labels, z_labels
for i in range(len(x_labels)):
    ax18.text(x_labels[i], y_labels[i], z_labels[i], emos_index1[i], size=10, zorder=1, color='k')
ax18.set_xlabel('UMAP_C1')
ax18.set_ylabel('UMAP_C2')
ax18.set_zlabel('UMAP_C3')

#Let's also save this
plt.savefig('participant'+str(pin)+'embed_torus-or-sphere-with-emos_lowres.png', dpi=300)
plt.savefig(input+'participant_'+pin+'_umap_with_emolabels_lowres.png')
#%%
#Let's plot the intermediate embedding in 3D
#Let's find the coordinates of the label points in the intermediate embedding
x_labels = interm_embedding.embedding_[emo_indices,0]
y_labels = interm_embedding.embedding_[emo_indices,1]
z_labels = interm_embedding.embedding_[emo_indices,2]

fig17 = plt.figure()
#Let's make the figure size 800x800
fig17.set_size_inches(8,8)
ax17 = fig17.add_subplot(111, projection='3d')

ax17.scatter(interm_embedding.embedding_[:, 0], interm_embedding.embedding_[:, 1], interm_embedding.embedding_[:, 2], cmap='viridis', s=0.1)
#Let's add the emotion labels to the plot
# for i in range(len(x_labels)):
#    ax17.text(interm_embedding.embedding_[i,0], interm_embedding.embedding_[i,1], interm_embedding.embedding_[i,2], emos_index1[i], size=14, zorder=1, color='k')
ax17.set_xlabel('UMAP_C1')
ax17.set_ylabel('UMAP_C2')
ax17.set_zlabel('UMAP_C3')

#Let's also save this
plt.savefig('participant'+str(pin)+'embed_naive_lowres.png', dpi=300)
#%%
## %%
#Let's also try embedding pca_results_with_emos_concat onto a sphere with UMAP
sphere_mapper_with_emos = umap.UMAP(output_metric='haversine', n_neighbors = 3).fit(cont_data_final1)
sphere_embedding_with_emos = sphere_mapper_with_emos.embedding_
##%%`
# Let's now plot the result of UMAP
# Let's first get the coordinates of the points on the sphere
# We will use the formula x = r*sin(theta)*cos(phi), y = r*sin(theta)*sin(phi), z = r*cos(theta)
# theta = sphere_mapper_with_emos.embedding_[:,0], phi = sphere_mapper_with_emos.embedding_[:,1], r = 1

x_sphere = np.sin(sphere_embedding_with_emos[:,0])*np.cos(sphere_embedding_with_emos[:,1])
y_sphere = np.sin(sphere_embedding_with_emos[:,0])*np.sin(sphere_embedding_with_emos[:,1])
z_sphere = np.cos(sphere_embedding_with_emos[:,0])

##%%
#Let's now plot the labels on the sphere
#We also want to add to the plot the labels that correspond to the emotion words
#The indices of rows that correspond to our emotion words, so they follow a pattern of i*len(resampled_data[0]),i*len(resampled_data[0]) + len(resampled_data[0]) for i = 0,1,2,...
x_emo_labels = x_sphere[emo_indices]
y_emo_labels = y_sphere[emo_indices]
z_emo_labels = z_sphere[emo_indices]

##%%
#Let's plot this as a figure with a big enough size, say, 800x800
fig19 = plt.figure()
#Let's make the figure size 800x800
fig19.set_size_inches(8,8)
ax19 = fig19.add_subplot(111, projection='3d')

ax19.scatter(x_sphere, y_sphere, z_sphere, cmap='viridis', s=0.1)
#We now want to add the labels to the plot at the corresponding coordinates of x_labels, y_labels, z_labels
for i in range(len(x_emo_labels)):
    ax19.text(x_emo_labels[i], y_emo_labels[i], z_emo_labels[i], emos_index1[i], size=10, zorder=1, color='k')
ax19.set_xlabel('UMAP_C1')
ax19.set_ylabel('UMAP_C2')
ax19.set_zlabel('UMAP_C3')
##%%
#Let's save this fig19 as a png, and then save the torus_mapper_emos.embedding_ and sphere_mapper_with_emos.embedding_ as csv files
fig19.savefig('participant'+str(pin)+'embed_sphere-with-emos.png', dpi=300)
fig18.savefig('participant'+str(pin)+'embed_torus-with-emos.png', dpi=300)
#Let's pickle torus_mapper_emos.embedding_ and sphere_mapper_with_emos.embedding_
np.savetxt('participant'+str(pin)+'embed_torus-with-emos.csv', final_embedding.embedding_, delimiter=',')
np.savetxt('participant'+str(pin)+'embed_sphere-with-emos.csv', sphere_mapper_with_emos.embedding_, delimiter=',')
#%%

#Let's try to analyze the embedding and find a geodesic path between two points on the sphere

#Let's first find the indices of the points on the sphere which have the same label in emos
#Let's compare first_labels and last_labels and take the index of the points which have identical values in both

index_ident = []
for i in range(len(first_labels)):
    for j in range(len(last_labels)):
        if first_labels[i] == last_labels[j] and i == j:
            index_ident.append((i,j))
# %%
#Now let's take the indices of the two points on the sphere by finding the index of those labels with emo_indices

indices_points = [emo_indices[index_ident[0][0]*2],emo_indices[index_ident[0][0]*2+1]]
# %%
#Now let's find the geodesic path between those two points
#We will use the formula for the geodesic path on the sphere
#Let's load the embedding
sphere_embedding_with_emos = np.loadtxt('participant'+str(pin)+'embed_sphere-with-emos.csv', delimiter=',')
x_sphere = np.sin(sphere_embedding_with_emos[:,0])*np.cos(sphere_embedding_with_emos[:,1])
y_sphere = np.sin(sphere_embedding_with_emos[:,0])*np.sin(sphere_embedding_with_emos[:,1])
z_sphere = np.cos(sphere_embedding_with_emos[:,0])
#Let's first find the coordinates of the two points
x1 = x_sphere[indices_points[0]]
y1 = y_sphere[indices_points[0]]
z1 = z_sphere[indices_points[0]]
x2 = x_sphere[indices_points[1]]
y2 = y_sphere[indices_points[1]]
z2 = z_sphere[indices_points[1]]

#Let's find the geodesic path between the two points
#We will use the formula for the geodesic path on the sphere
geodesic_path = np.arccos(x1*x2 + y1*y2 + z1*z2)

#%%
#Let's plot the geodesic path on the sphere
fig20 = plt.figure()
#Let's make the figure size 800x800
fig20.set_size_inches(8,8)
ax20 = fig20.add_subplot(111, projection='3d')

ax20.scatter(x_sphere, y_sphere, z_sphere, cmap='viridis', s=0.1)
#We now want to add the labels to the plot at the corresponding coordinates of x_labels, y_labels, z_labels
#for i in range(len(x_emo_labels)):
#    ax20.text(x_emo_labels[i], y_emo_labels[i], z_emo_labels[i], emos_index1[i], size=10, zorder=1, color='k')
ax20.set_xlabel('UMAP_C1')
ax20.set_ylabel('UMAP_C2')
ax20.set_zlabel('UMAP_C3')
#Let's also plot the geodesic path
ax20.plot([x1,x2],[y1,y2],[z1,z2], color='r')
#Let's save this fig20 as a png
fig20.savefig('participant'+str(pin)+'embed_sphere-with-emos_with_geodesic_path.png', dpi=300)
fig20.show()
# %%
import matplotlib 
print('The versions of numpy, pandas, and UMAP used in this analysis were: \n numpy: '+np.__version__+
        '\n pandas: '+pd.__version__+
        '\n UMAP: '+umap.__version__+
        '\n matplotlib: '+matplotlib.__version__)

#%%
#Let's also print the version of matplotlib.pyplot
