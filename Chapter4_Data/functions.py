#%% In this script we will define functions that will pull out the data from our input file

#Let's start with importing pandas and numpy, and then we will import our data file

import pandas as pd
import numpy as np

#Here let's define a function that will pull out each individual participant's data

def pull_participant(log_full, pin):
    '''
    This function will pull out the data for each participant.
    It requires two input parameters:
    log_full - the full log file presented as a DataFrame (it has to have the 'ExternalReference' column)
    pin - the PIN number for the participant
    '''
    participant_logs = pd.DataFrame()
    # Iterate over the rows of the 'log_full' DataFrame
    for row in log_full.iterrows():
        # If the 'ExternalReference' column of the current row is equal to the random value
        if row[1]['ExternalReference'] == pin:
            # Add the current row to the 'participant_logs' DataFrame
            participant_logs = pd.concat([participant_logs, row[1].to_frame().transpose()])
    return participant_logs
#%% Now we will define a function that will pull out the data from the log file for one participant

def pull_data_bio_1(participant_logs):
    participant_id_bio = pd.DataFrame()

    for row1 in participant_logs.iterrows():
        bio1 = row1[1][['Q1_9','Q1_10','Q2',
                    'BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
                    'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
                    'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
                    'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']]
        bio1 = bio1.to_frame()
        bio1 = bio1.transpose()
        participant_id_bio = pd.concat([participant_id_bio, bio1], axis = 0)

        if row1[1]['Q0'] == 'Yes':
            bio2 = row1[1][['Q1_9.1','Q1_10.1','Q2.1',
                            'BC2-1_1','BC2-1_2','BC2-1_3','BC2-1_4','BC2-1_5',
                            'BC2-1_6','BC2-1_7','BC2-1_8','BC2-1_9','BC2-1_10',
                            'BC2-1_11','BC2-1_12','BC2-1_13','BC2-1_14','BC2-1_15',
                            'BC2-1_16','BC2-1_17','BC2-1_18','BC2-1_19','BC2-1_20']]
            bio2 = bio2.to_frame()
            bio2 = bio2.transpose()
            bio2.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
                            'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
                            'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
                            'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
            participant_id_bio = pd.concat([participant_id_bio, bio2],axis = 0)
        
        if row1[1]['Q0.1'] == 'Yes':
            bio3 = row1[1][['Q1_9.2','Q1_10.2','Q2.2',
                            'BC3-1_1','BC3-1_2','BC3-1_3','BC3-1_4','BC3-1_5',
                            'BC3-1_6','BC3-1_7','BC3-1_8','BC3-1_9','BC3-1_10',
                            'BC3-1_11','BC3-1_12','BC3-1_13','BC3-1_14','BC3-1_15',
                            'BC3-1_16','BC3-1_17','BC3-1_18','BC3-1_19','BC3-1_20']]
            bio3 = bio3.to_frame()
            bio3 = bio3.transpose()
            bio3.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
                            'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
                            'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
                            'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
            participant_id_bio = pd.concat([participant_id_bio, bio3],axis = 0)

        if row1[1]['Q0.2'] == 'Yes':
            bio4 = row1[1][['Q1_9.3','Q1_10.3','Q2.3',
                            'BC4-1_1','BC4-1_2','BC4-1_3','BC4-1_4','BC4-1_5',
                            'BC4-1_6','BC4-1_7','BC4-1_8','BC4-1_9','BC4-1_10',
                            'BC4-1_11','BC4-1_12','BC4-1_13','BC4-1_14','BC4-1_15',
                            'BC4-1_16','BC4-1_17','BC4-1_18','BC4-1_19','BC4-1_20']]
            bio4 = bio4.to_frame()
            bio4 = bio4.transpose()
            bio4.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
                            'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
                            'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
                            'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
            participant_id_bio = pd.concat([participant_id_bio, bio4],axis = 0)

        if row1[1]['Q51'] == 'Yes':
            bio5 = row1[1][['Q52_9','Q52_10','Q53',
                            'Q54_1','Q54_2','Q54_3','Q54_4','Q54_5',
                            'Q54_6','Q54_7','Q54_8','Q54_9','Q54_10',
                            'Q54_11','Q54_12','Q54_13','Q54_14','Q54_15',
                            'Q54_16','Q54_17','Q54_18','Q54_19','Q54_20']]
            bio5 = bio5.to_frame()
            bio5 = bio5.transpose()
            bio5.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
                            'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
                            'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
                            'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
            participant_id_bio = pd.concat([participant_id_bio, bio5])

    return participant_id_bio

#%%
def pull_data_bio_2(participant_logs):
    participant_id_bio = pd.DataFrame()

    for row1 in participant_logs.iterrows():
        bio1 = row1[1][['Q1_9','Q1_10','Q2','BC1-2_1','BC1-2_2','BC1-2_3','BC1-2_4','BC1-2_5',
                    'BC1-2_6','BC1-2_7']]
        bio1 = bio1.to_frame()
        bio1 = bio1.transpose()
        participant_id_bio = pd.concat([participant_id_bio,bio1])

        if row1[1]['Q0'] == 'Yes':
            bio2 = row1[1][['Q1_9.1','Q1_10.1','Q2.1',
                            'BC2-2_1','BC2-2_2','BC2-2_3','BC2-2_4','BC2-2_5',
                            'BC2-2_6','BC2-2_7']]
            bio2 = bio2.to_frame()
            bio2 = bio2.transpose()
            bio2.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-2_1','BC1-2_2','BC1-2_3','BC1-2_4','BC1-2_5',
                            'BC1-2_6','BC1-2_7']
            participant_id_bio = pd.concat([participant_id_bio,bio2])
        
        if row1[1]['Q0.1'] == 'Yes':
            bio3 = row1[1][['Q1_9.2','Q1_10.2','Q2.2',
                            'BC3-2_1','BC3-2_2','BC3-2_3','BC3-2_4','BC3-2_5',
                            'BC3-2_6','BC3-2_7']]
            bio3 = bio3.to_frame()
            bio3 = bio3.transpose()
            bio3.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-2_1','BC1-2_2','BC1-2_3','BC1-2_4','BC1-2_5',
                            'BC1-2_6','BC1-2_7']
            participant_id_bio = pd.concat([participant_id_bio,bio3])

        if row1[1]['Q0.2'] == 'Yes':
            bio4 = row1[1][['Q1_9.3','Q1_10.3','Q2.3',
                            'BC4-2_1','BC4-2_2','BC4-2_3','BC4-2_4','BC4-2_5',
                            'BC4-2_6','BC4-2_7']]
            bio4 = bio4.to_frame()
            bio4 = bio4.transpose()
            bio4.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-2_1','BC1-2_2','BC1-2_3','BC1-2_4','BC1-2_5',
                            'BC1-2_6','BC1-2_7']
            participant_id_bio = pd.concat([participant_id_bio,bio4])

        if row1[1]['Q51'] == 'Yes':
            bio5 = row1[1][['Q52_9','Q52_10','Q53',
                            'Q55_1','Q55_2','Q55_3','Q55_4','Q55_5',
                            'Q55_6','Q55_7']]
            bio5 = bio5.to_frame()
            bio5 = bio5.transpose()
            bio5.columns = ['Q1_9','Q1_10','Q2',
                            'BC1-2_1','BC1-2_2','BC1-2_3','BC1-2_4','BC1-2_5',
                            'BC1-2_6','BC1-2_7']
            participant_id_bio = pd.concat([participant_id_bio,bio5])

    return participant_id_bio

# %% Now let's write a function that will pull out the psychological dimension data for one participant

def pull_data_psych(participant_logs):
    participant_id_psy = pd.DataFrame()

    for row2 in participant_logs.iterrows():
        psy1 = row2[1].loc[['Q1_9','Q1_10','Q2',
                            'TM1_1','TM1_2','TM1_3','TM1_4','TM1_5',
                            'TM1_6','TM1_7','TM1_8','TM1_9','TM1_10',
                            'TM1_11','TM1_12','TM1_13','TM1_14','TM1_15',
                            'TM1_16','TM1_17','TM1_18','TM1_19','TM1_20']]
        psy1 = psy1.to_frame()
        psy1 = psy1.transpose()
        participant_id_psy = pd.concat([participant_id_psy, psy1])

        if row2[1].loc['Q0'] == 'Yes':
            psy2 = row2[1].loc[['Q1_9.1','Q1_10.1','Q2.1',
                                'TM2_1','TM2_2','TM2_3','TM2_4','TM2_5',
                                'TM2_6','TM2_7','TM2_8','TM2_9','TM2_10',
                                'TM2_11','TM2_12','TM2_13','TM2_14','TM2_15',
                                'TM2_16','TM2_17','TM2_18','TM2_19','TM2_20']]
            psy2 = psy2.to_frame()
            psy2 = psy2.transpose()
            psy2.columns = ['Q1_9','Q1_10','Q2',
                            'TM1_1','TM1_2','TM1_3','TM1_4','TM1_5',
                            'TM1_6','TM1_7','TM1_8','TM1_9','TM1_10',
                            'TM1_11','TM1_12','TM1_13','TM1_14','TM1_15',
                            'TM1_16','TM1_17','TM1_18','TM1_19','TM1_20']
            participant_id_psy = pd.concat([participant_id_psy, psy2])
        
        if row2[1].loc['Q0.1'] == 'Yes':
            psy3 = row2[1].loc[['Q1_9.2','Q1_10.2','Q2.2',
                                'TM3_1','TM3_2','TM3_3','TM3_4','TM3_5',
                                'TM3_6','TM3_7','TM3_8','TM3_9','TM3_10',
                                'TM3_11','TM3_12','TM3_13','TM3_14','TM3_15',
                                'TM3_16','TM3_17','TM3_18','TM3_19','TM3_20']]
            psy3 = psy3.to_frame()
            psy3 = psy3.transpose()
            psy3.columns = ['Q1_9','Q1_10','Q2',
                            'TM1_1','TM1_2','TM1_3','TM1_4','TM1_5',
                            'TM1_6','TM1_7','TM1_8','TM1_9','TM1_10',
                            'TM1_11','TM1_12','TM1_13','TM1_14','TM1_15',
                            'TM1_16','TM1_17','TM1_18','TM1_19','TM1_20']
            participant_id_psy = pd.concat([participant_id_psy, psy3])

        if row2[1].loc['Q0.2'] == 'Yes':
            psy4 = row2[1].loc[['Q1_9.3','Q1_10.3','Q2.3',
                                'TM4_1','TM4_2','TM4_3','TM4_4','TM4_5',
                                'TM4_6','TM4_7','TM4_8','TM4_9','TM4_10',
                                'TM4_11','TM4_12','TM4_13','TM4_14','TM4_15',
                                'TM4_16','TM4_17','TM4_18','TM4_19','TM4_20']]
            psy4 = psy4.to_frame()
            psy4 = psy4.transpose()
            psy4.columns = ['Q1_9','Q1_10','Q2',
                            'TM1_1','TM1_2','TM1_3','TM1_4','TM1_5',
                            'TM1_6','TM1_7','TM1_8','TM1_9','TM1_10',
                            'TM1_11','TM1_12','TM1_13','TM1_14','TM1_15',
                            'TM1_16','TM1_17','TM1_18','TM1_19','TM1_20']
            participant_id_psy = pd.concat([participant_id_psy, psy4])

        if row2[1].loc['Q51'] == 'Yes':
            psy5 = row2[1].loc[['Q52_9','Q52_10','Q53',
                                'Q56_1','Q56_2','Q56_3','Q56_4','Q56_5',
                                'Q56_6','Q56_7','Q56_8','Q56_9','Q56_10',
                                'Q56_11','Q56_12','Q56_13','Q56_14','Q56_15',
                                'Q56_16','Q56_17','Q56_18','Q56_19','Q56_20']]
            psy5 = psy5.to_frame()
            psy5 = psy5.transpose()
            psy5.columns = ['Q1_9','Q1_10','Q2',
                            'TM1_1','TM1_2','TM1_3','TM1_4','TM1_5',
                            'TM1_6','TM1_7','TM1_8','TM1_9','TM1_10',
                            'TM1_11','TM1_12','TM1_13','TM1_14','TM1_15',
                            'TM1_16','TM1_17','TM1_18','TM1_19','TM1_20']
            participant_id_psy = pd.concat([participant_id_psy, psy5])

    return participant_id_psy

#%% Now let's write a function for social dimension data

def pull_data_soc(participant_logs):
    participant_id_soc = pd.DataFrame()

    for row3 in participant_logs.iterrows():
        soc1 = row3[1].loc[['Q1_9','Q1_10','Q2',
                            'SC1_1','SC1_2','SC1_3','SC1_4','SC1_5',
                            'SC1_6','SC1_7','SC1_8','SC1_9','SC1_10',
                            'SC1_11','SC1_12','SC1_13','SC1_14','SC1_15',
                            'SC1_16','SC1_17','SC1_18','SC1_19','SC1_20']]
        soc1 = soc1.to_frame()
        soc1 = soc1.transpose()
        participant_id_soc = pd.concat([participant_id_soc, soc1])

        if row3[1].loc['Q0'] == 'Yes':
            soc2 = row3[1].loc[['Q1_9.1','Q1_10.1','Q2.1',
                                'SC2_1','SC2_2','SC2_3','SC2_4','SC2_5',
                                'SC2_6','SC2_7','SC2_8','SC2_9','SC2_10',
                                'SC2_11','SC2_12','SC2_13','SC2_14','SC2_15',
                                'SC2_16','SC2_17','SC2_18','SC2_19','SC2_20']]
            soc2 = soc2.to_frame()
            soc2 = soc2.transpose()
            soc2.columns = ['Q1_9','Q1_10','Q2',
                            'SC1_1','SC1_2','SC1_3','SC1_4','SC1_5',
                            'SC1_6','SC1_7','SC1_8','SC1_9','SC1_10',
                            'SC1_11','SC1_12','SC1_13','SC1_14','SC1_15',
                            'SC1_16','SC1_17','SC1_18','SC1_19','SC1_20']
            participant_id_soc = pd.concat([participant_id_soc, soc2])
        
        if row3[1].loc['Q0.1'] == 'Yes':
            soc4 = row3[1].loc[['Q1_9.2','Q1_10.2','Q2.2',
                                'SC3_1','SC3_2','SC3_3','SC3_4','SC3_5',
                                'SC3_6','SC3_7','SC3_8','SC3_9','SC3_10',
                                'SC3_11','SC3_12','SC3_13','SC3_14','SC3_15',
                                'SC3_16','SC3_17','SC3_18','SC3_19','SC3_20']]
            soc4 = soc4.to_frame()
            soc4 = soc4.transpose()
            soc4.columns = ['Q1_9','Q1_10','Q2',
                            'SC1_1','SC1_2','SC1_3','SC1_4','SC1_5',
                            'SC1_6','SC1_7','SC1_8','SC1_9','SC1_10',
                            'SC1_11','SC1_12','SC1_13','SC1_14','SC1_15',
                            'SC1_16','SC1_17','SC1_18','SC1_19','SC1_20']
            participant_id_soc = pd.concat([participant_id_soc, soc4])

        if row3[1].loc['Q0.2'] == 'Yes':
            soc5 = row3[1].loc[['Q1_9.3','Q1_10.3','Q2.3',
                                'SC4_1','SC4_2','SC4_3','SC4_4','SC4_5',
                                'SC4_6','SC4_7','SC4_8','SC4_9','SC4_10',
                                'SC4_11','SC4_12','SC4_13','SC4_14','SC4_15',
                                'SC4_16','SC4_17','SC4_18','SC4_19','SC4_20']]
            soc5 = soc5.to_frame()
            soc5 = soc5.transpose()
            soc5.columns = ['Q1_9','Q1_10','Q2',
                            'SC1_1','SC1_2','SC1_3','SC1_4','SC1_5',
                            'SC1_6','SC1_7','SC1_8','SC1_9','SC1_10',
                            'SC1_11','SC1_12','SC1_13','SC1_14','SC1_15',
                            'SC1_16','SC1_17','SC1_18','SC1_19','SC1_20']
            participant_id_soc = pd.concat([participant_id_soc, soc5])

        if row3[1].loc['Q51'] == 'Yes':
            soc6 = row3[1].loc[['Q52_9','Q52_10','Q53',
                                'Q57_1','Q57_2','Q57_3','Q57_4','Q57_5',
                                'Q57_6','Q57_7','Q57_8','Q57_9','Q57_10',
                                'Q57_11','Q57_12','Q57_13','Q57_14','Q57_15',
                                'Q57_16','Q57_17','Q57_18','Q57_19','Q57_20']]
            soc6 = soc6.to_frame()
            soc6 = soc6.transpose()
            soc6.columns = ['Q1_9','Q1_10','Q2',
                            'SC1_1','SC1_2','SC1_3','SC1_4','SC1_5',
                            'SC1_6','SC1_7','SC1_8','SC1_9','SC1_10',
                            'SC1_11','SC1_12','SC1_13','SC1_14','SC1_15',
                            'SC1_16','SC1_17','SC1_18','SC1_19','SC1_20']
            participant_id_soc = pd.concat([participant_id_soc, soc6])

    return participant_id_soc

            
# %% Let's write a function that will interpolate the data for all three dimensions and pull them into one dataframe

def interpolate_data(log_full, participant_id, dimension, index, emos_return = False, fig_return = False):
    '''
    This function will interpolate the data for each participant and pull them into one dataframe.
    Input will be the PIN number for the participant.
    Output will be the dataframe with the interpolated data for all three dimensions.
    
    The function pulls in a previous function defined by the dimension of the data.
    
    '''

    participant = pull_participant(log_full, participant_id)

    if dimension == 'bio':
        to_be_interpolated = pull_data_bio_1(participant)
        columns = ['BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
                   'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
                   'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
                   'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
    elif dimension == 'psy':
        to_be_interpolated = pull_data_psych(participant)
        columns = ['TM1_1','TM1_2','TM1_3','TM1_4','TM1_5',
                   'TM1_6','TM1_7','TM1_8','TM1_9','TM1_10',
                   'TM1_11','TM1_12','TM1_13','TM1_14','TM1_15',
                   'TM1_16','TM1_17','TM1_18','TM1_19','TM1_20']
    elif dimension == 'soc':
        to_be_interpolated = pull_data_soc(participant)
        columns = ['SC1_1','SC1_2','SC1_3','SC1_4','SC1_5',
                   'SC1_6','SC1_7','SC1_8','SC1_9','SC1_10',
                   'SC1_11','SC1_12','SC1_13','SC1_14','SC1_15',
                   'SC1_16','SC1_17','SC1_18','SC1_19','SC1_20']

    dict_interp = {'A few seconds':1, 
                   'Several minutes to half an hour':60, 
                   'Half an hour to a few hours':180, 
                   'Half a day or more':540}
    dur = to_be_interpolated.iloc[index]['Q2']
    if emos_return == True:
        emos = [to_be_interpolated.iloc[index]['Q1_9'], to_be_interpolated.iloc[index]['Q1_10']]
    interp_value = dict_interp[dur]

    interp1 = to_be_interpolated.iloc[index][columns]
    #Let's convert zeros in the data that are actually missing values to pandas nan
    interp1 = interp1.replace('0', np.nan)
    interp1 = interp1.astype(float)
    #Let's also drop all non-nan values that are surrounded by nan as they are invalid
    #We will check for interp.iloc[i-1] and interp.iloc[i+1] to be nan
    for i in range(1,len(interp1)-1):
        if np.isnan(interp1.iloc[i-1]) and np.isnan(interp1.iloc[i+1]):
            interp1.iloc[i] = np.nan
    interp1 = interp1.dropna()
    #Let's also convert all values in interp1 to int
    interp1 = interp1.astype(int)
    #Now let's check that if interp_value is more than 1, the length is more than 2 and interpolation can be done
    if interp_value >= 1 and len(interp1) > 2:
        interp2 = pd.Series(np.nan, index = np.arange(len(interp1)*interp_value))

        for i in range(len(interp1)):
            interp2[i*interp_value] = interp1[i]

        interp3 = interp2.interpolate(method = 'quadratic')
        interp3 = interp3.dropna()
    #If the length of interp1 is less than 2, then we will just return the original data and print a warning
    else:
        interp3 = interp1
        print('Warning: Not enough data to interpolate')
    #Let's add a figure with the plot of the interpolated data
    if fig_return:
        fig = interp3.plot()
        if emos_return:
            return interp3,emos, fig
        else:
            return interp3, fig
    else:
        if emos_return:
            return interp3, emos
        else:
            return interp3
# %%
#%% Let's try writing a function for stretching the data across three dimensions to match the longest dimension

def stretch_and_interp(shorter,longer):
    '''
    This generic function will stretch the shorter dimension to match the length of the longest dimension,
    and then interpolate any missing values
    
    '''
    shorter_temp = np.repeat(np.nan,len(longer))
    diff_int = len(longer) / len(shorter)
    for i in range(len(shorter)):
        x = round(i*diff_int)
        shorter_temp[x] = shorter[i]
    shorter_temp = pd.Series(shorter_temp)
    shorter = shorter_temp.interpolate(method = 'quadratic')
    return shorter

# %% Now let's write a function to stretch and interpolate the data for all three dimensions, and then plot them in 3d
def equate_dims(dim1, dim2, dim3):
    from functions import stretch_and_interp
    longest = max(len(dim1), len(dim2), len(dim3))
    if len(dim1) == longest:
        dim1n = dim1
        dim2n = stretch_and_interp(dim2,dim1)
        dim3n = stretch_and_interp(dim3,dim1)
    elif len(dim2) == longest:
        dim2n = dim2
        dim1n = stretch_and_interp(dim1,dim2)
        dim3n = stretch_and_interp(dim3,dim2)
    elif len(dim3) == longest:
        dim3n = dim3
        dim1n = stretch_and_interp(dim1,dim3)
        dim2n = stretch_and_interp(dim2,dim3)
    return dim1n, dim2n, dim3n

#%%
#Let's try and debug the equate_dims function

# log_full = pd.read_csv('/N/slate/ksalibay/Behavioral_Data/Data_Nov23/log_20231021.csv')
# pin = '76697'
# participant = pull_participant(log_full, pin)
# bio = pull_data_bio_1(participant)
# psy = pull_data_psych(participant)
# soc = pull_data_soc(participant)

# # # #%%
# # # columns = ['BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
# # #                    'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
# # #                    'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
# # #                    'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
# # # row_temp = bio.iloc[12][columns]
# # # row_temp = row_temp.astype(float)
# # # for i in range(3,len(row_temp)-1):
# # #         if np.isnan(row_temp.iloc[i-1]) and np.isnan(row_temp.iloc[i+1]):
# # #             row_temp.iloc[i] = np.nan
# # # #%%
# row1, emos = interpolate_data(log_full, pin, 'bio', 13, emos_return = True)
#%%

# #For some reason, the interpolate_data function is not working for the bio dimension
# #Let's try and debug that
# #%%
# #Let's try and debug the interpolate_data function for the bio dimension
# participant = pull_participant(log_full, pin)
# to_be_interpolated = pull_data_bio_1(participant)
# dict_interp = {'A few seconds':1,
#                 'Several minutes to half an hour':900,
#                 'Half an hour to a few hours':21600,
#                 'Half a day or more':43200}
# dur = to_be_interpolated.iloc[0]['Q2']
# interp_value = dict_interp[dur]
# print(to_be_interpolated.iloc[0])
# #%%

# columns = ['BC1-1_1','BC1-1_2','BC1-1_3','BC1-1_4','BC1-1_5',
#             'BC1-1_6','BC1-1_7','BC1-1_8','BC1-1_9','BC1-1_10',
#             'BC1-1_11','BC1-1_12','BC1-1_13','BC1-1_14','BC1-1_15',
#             'BC1-1_16','BC1-1_17','BC1-1_18','BC1-1_19','BC1-1_20']
# interp1 = to_be_interpolated.iloc[0][columns]

# #%%
# print(interp1)
# interp1 = interp1.replace('0', np.nan)
# print(interp1)
# #%%
# interp1 = interp1.dropna()
# print(interp1)
# #%%
# interp2 = pd.Series(np.nan, index = np.arange(len(interp1)*interp_value))
# interp2 = interp2.interpolate(method = 'quadratic')

# #%%

# #The output of row1 and interp2 are different. Let's try and debug that
# row1 = interpolate_data(log_full, pin, 'bio', 1)
# #For some reason, the interpolation does not happen when running the function, 
# #but it does happen when running the lines of code individually
# #Let's try and debug that

# #%%
# row2 = interpolate_data(log_full, pin, 'psy', 0)
# row3 = interpolate_data(log_full, pin, 'soc', 0)

# #%%
# #Let's try running the equate_dims function line by line
# longest = max(len(row1), len(row2), len(row3))
# if len(row1) == longest:
#     print('row1 is the longest')
# elif len(row2) == longest:
#     print('row2 is the longest')
# elif len(row3) == longest:
#     print('row3 is the longest')

# #%%
# row3n = row3
# #Okay, so the error is in the stretch_and_interp function. Let's try and debug that
# shorter = row2
# longer = row3
# shorter_temp = np.repeat(np.nan,len(longer))
# diff_int = len(longer) / len(shorter)
# #%%
# for i in range(len(shorter)):
#     x = round(i*diff_int)
#     shorter_temp[x] = shorter[i]

#%%
def make_3d(dim1, dim2, dim3, emos):
    '''
    This function takes in the results of the interpolate_data function for all three dimensions,
    stretches them to match the longest dimension, interpolates, and then plots them in a 3D plot.

    dim1, dim2, and dim3 are all the results of the interpolate_data function for each dimension.
    emos is a list of the two emotions that the participant chose and is the output of one of the interpolate_data functions.
    '''
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig = ax.scatter3D(dim1, dim2, dim3, c=dim1, cmap='Greens')
    #fig.save(pin1+'1_3Dplot.png')
    ax.text(dim1.iloc[0], dim2.iloc[0], dim3.iloc[0], emos[0], color='red')
    ax.text(dim1.iloc[-1], dim2.iloc[-1], dim3.iloc[-1], emos[1], color='red')

    return fig
