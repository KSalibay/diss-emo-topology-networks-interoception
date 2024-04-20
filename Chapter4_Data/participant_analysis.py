#%%
#import functions
import pandas as pd
import numpy as np

input = '/N/slate/ksalibay/Behavioral_Data/Data_Nov23/'
fname = 'log_20231021.csv'

log_full = pd.read_csv(input+fname)

## %% Let's get the list of participants' PINs
pins = log_full['ExternalReference'].unique()
pins = pd.Series(pins)
pins = pins.dropna()
pins = pins[2:]
pins = pins[pins != '1']

##%% Here let's get the participant we are interested in by their PIN
#To do that, let's look at the interesting cases where 'Q51' is equal to 'Yes' and list those 'ExternalReference' values.

five_yes = log_full[log_full['Q51'] == 'Yes']
five_yes = five_yes['ExternalReference'].unique()

#%%

four_yes = log_full[log_full['Q0.2'] == 'Yes']
four_yes = four_yes['ExternalReference'].unique()

#Let's only retain the values in four_yes that are not in five_yes
four_yes = [x for x in four_yes if x not in five_yes]

#%%

#Let's look through the pins where Q0.1 is equal to 'Yes'
three_yes = log_full[log_full['Q0.1'] == 'Yes']
three_yes = three_yes['ExternalReference'].unique()

#Now let's look through the file of dropped participants and drop them from three_yes
dropped = pd.read_csv(input+'dropped_participants.csv')
dropped = [str(x) for x in dropped.iloc[:,0]]
three_yes = [x for x in three_yes if x not in dropped]

#Let's also cross-check with four_yes and drop those that are in four_yes and same for five_yes
three_yes = [x for x in three_yes if x not in four_yes]
three_yes = [x for x in three_yes if x not in five_yes]
#%%
#Now let's check two_yes as a function of responses to log_full['Q0'],
#and then cross-check with three_yes and leave only those which are not in three_yes currently

two_yes = log_full[log_full['Q0'] == 'Yes']
two_yes = two_yes['ExternalReference'].unique()
two_yes = [x for x in two_yes if x not in three_yes]
two_yes = [x for x in two_yes if x not in four_yes]
two_yes = [x for x in two_yes if x not in five_yes]
#error_index = two_yes.index('54608')
#two_yes = two_yes[error_index+1:]
#%%
one_yes = ['23554', '91339', '502', '4182', '159', '53512', '41975', '89263', '60341', '62766', '17747', 
           '28903', '52913', '88520', '68674', '56571', '25206', '93465', '96921', '9087', '87753', '96139', 
           '32744', '1421', '32603', '53352', '27406', '62409', '44576', '23398', '4528', '96541', '95505']
#%% Let's get the participant we are interested in by their PIN
#We have 23 values in three_yes, so let's go through them one by one
pin1 = one_yes[32]
import functions 
participant = functions.pull_participant(log_full, pin1)

participant_bio = functions.pull_data_bio_1(participant)
participant_psy = functions.pull_data_psych(participant)
participant_soc = functions.pull_data_soc(participant)

##%% Now let's try running the above code for all the rows in one participant's data
interp_bio = []
interp_psy = []
interp_soc = []
emos_index = []

#Let's loop through each row of participant_bio and interpolate the data for each row
#Let's create the range for just the rows where length is greater than 3
   
for i in range(len(participant_bio)):
    row1, emos = functions.interpolate_data(log_full, pin1, 'bio', i, emos_return = True)
    row2 = functions.interpolate_data(log_full, pin1, 'psy', i)
    row3 = functions.interpolate_data(log_full, pin1, 'soc', i)
    #Let's check if the len of any of row1, row2, row3 is 0 and drop them then, breaking the loop
    if len(row1) == 0 or len(row2) == 0 or len(row3) == 0:
        print('dropping row '+str(i))
        continue
    #Let's also drop the row if the length of any of row1, row2, row3 is less than 3
    elif len(row1) < 3 or len(row2) < 3 or len(row3) < 3:
        print('dropping row '+str(i))
        continue
    else:
        newbio, newpsy, newsoc = functions.equate_dims(row1, row2, row3)
        interp_bio.append(newbio)
        interp_psy.append(newpsy)
        interp_soc.append(newsoc)
        emos_index.append(emos)

print(len(interp_bio))
##%%
#Since this participant's data is so short and not very interesting, we will drop them.
#Let's create a new csv file and drop pin1 into it.
#Later, when we are running through the entire file again and dropping this variable, 
#we will need to only append new values to this file, and not overwrite it.
if len(interp_bio) < 10:
    dropped = pd.DataFrame()
    dropped['PIN ID'] = [pin1]
    dropped.to_csv(input+'dropped_participants.csv', mode='a', header=False, index=False)
    print('This participant '+pin1+' has been dropped.')
else:
    print('This participant '+pin1+' has been kept.')

## %%

from random import randint
i = randint(0, len(interp_bio)-1)
fig = functions.make_3d(interp_bio[i], interp_psy[i], interp_soc[i], emos_index[i])

## %% Now let's import the rest of the participants' data (the 'static' dimensions)

fname_static = input+'init_test_cleaned.csv'
static = pd.read_csv(fname_static)

#Let's pull out only the row of static where 'PIN ID' is equal to pin1
part_static = static[static['PIN ID'] == float(pin1)]

##%% Let's stack interp_bio, interp_psy, interp_soc into one array as rows
#Let's write a loop that will go through each row of interp_bio, interp_psy, interp_soc and stack them into one array,
#then append that array to a new list called full_dynamic, and then also add in values equal to each column 
#of part_static repeated for the length of interp_bio[i]
full_dynamic = []
for i in range(len(interp_bio)):
    temp_array = np.empty((len(interp_bio[i]), 3), dtype=object)
    temp_array[:] = np.nan
    temp_array[:,0] = interp_bio[i]
    temp_array[:,1] = interp_psy[i]
    temp_array[:,2] = interp_soc[i]
    full_dynamic.append(temp_array.T)


## %% Let's save each element of full_dynamic as a csv file preserving the Pandas dataframe structure
#And let's assign the three columns of each csv file the names 'bio', 'psy', 'soc'
for i in range(len(full_dynamic)):
    temp_df = pd.DataFrame(full_dynamic[i].T, columns=['bio', 'psy', 'soc'])
    temp_df.to_csv(input+'participant_'+pin1+'_row_'+str(i)+'_lowres.csv', index=False)
#Let's also store the emos_index as a csv file
    emos_index = pd.DataFrame(emos_index)
    emos_index.to_csv(input+'participant_'+pin1+'_emos_index_lowres.csv', index=False)
#Now let's also save the static dimensions as a csv file
part_static.to_csv(input+'participant_'+pin1+'_static_dims_lowres.csv', index=False)
# %%
