#%% Importing libraries

import pandas as pd
import numpy as np

input = '/home/megumi/Documents/Studies/Summer 2023/Data/'
fname = 'init_test_20230911.csv'

init_test_data = pd.read_csv(input+fname)

fname2 = 'log_20230503.csv'

log_full = pd.read_csv(input+fname2)

#%% Cross-compare the cell with the unique ID and only keep the ones that are in both in init_test_data

init_test_data = init_test_data[init_test_data['PIN ID'].isin(log_full['ExternalReference'])]

#%% Anonymize the email addresses in Q5 in init_test_data by replacing them with "anonymized"

init_test_data['Q5'] = init_test_data['Q5'].str.replace(r'[^@]+@[^@]+\.[^@]+', 'anonymized')

#%% Extract only PIN ID, Q1_1-Q1_4, RQ1_5-RQ1_12, Q1_13-37, SRIS1_1-SRIS1_20, IRI1-IRI28

init_test_1 = init_test_data.iloc[:,22:]
# %% Now let's convert all the "Always" into 5 and "Never" into 1, then "Strongly agree" into 5 and "Strongly disagree" into 1, 
# #with "Slightly agree" into 4 and "Slightly disagree" into 2, and "Neither agree nor disagree" into 3, 
#and for the IRI questions, "Does not describe me well" into 1 and "Describes me very well" into 5.

init_test_2 = init_test_1.replace({'Always':5, 'Never':0,
                                   'Agree':5, 'Disagree':2,
                                   'Agree slightly':4, 'Disagree slightly':3,
                                   'Agree strongly':6, 'Disagree strongly':1,
                                   'Does not describe me well':1, 'Describes me very well':5})

#Let's get rid of NaNs and code them as 0s
init_test_2 = init_test_2.fillna(0)

#Let's convert all numerical values in columns 1-86 into integers

init_test_2 = init_test_2.iloc[:,0:86].astype(int)
#%% Conversion of MAIA
'''
Now that we have the data in the right format, we can start the analysis.
Let's start with MAIA-2 — columns Q1_1-Q1_4. RQ1_5-RQ1_12, Q1_13-37.
Out of those: Q1_1-Q1_4 correspond to "Noticing" pattern (take average of the four),
RQ1_5-RQ1_10 correspond to Not-Distracting scale (reverse the scores by subtracting from 6 and take average of the six),
RQ1_11-RQ1_12 and Q1_13-15 correspond to Not-Worrying (reverse RQ1_11, RQ1_12, RQ1_15 by subtracting from 6, 
    add to the sum of Q1_13-14 and take average of the five),
Q1_16-22 correspond to Attention Regulation (take average of the seven),
Q1_23-27 correspond to Emotional Awareness (take average of the five),
Q1_28-31 correspond to Self-Regulation (take average of the four),
Q1_32-34 correspond to Body Listening (take average of the three),
Q1_35-37 correspond to Trusting (take average of the three).

'''
#Let's create a new dataframe and put the averages there. Let's also pull out the score for Q1_32 separately.

noticing = init_test_2[['Q1_1', 'Q1_2', 'Q1_3', 'Q1_4']].mean(axis=1)
not_distracting = (5-init_test_2[['RQ1_5', 'RQ1_6', 'RQ1_7', 'RQ1_8', 'RQ1_9', 'RQ1_10']]).mean(axis=1)
not_worrying = (5-init_test_2[['RQ1_11', 'RQ1_12', 'RQ1_15']].mean(axis=1) + init_test_2[['Q1_13', 'Q1_14']].mean(axis=1))/2
attention_regulation = init_test_2[['Q1_16', 'Q1_17', 'Q1_18', 'Q1_19', 'Q1_20', 'Q1_21', 'Q1_22']].mean(axis=1)
emotional_awareness = init_test_2[['Q1_23', 'Q1_24', 'Q1_25', 'Q1_26', 'Q1_27']].mean(axis=1)
self_regulation = init_test_2[['Q1_28', 'Q1_29', 'Q1_30', 'Q1_31']].mean(axis=1)
body_listening = init_test_2[['Q1_32', 'Q1_33', 'Q1_34']].mean(axis=1)
trusting = init_test_2[['Q1_35', 'Q1_36', 'Q1_37']].mean(axis=1)
awareness = init_test_2['Q1_32']

init_test3 = pd.DataFrame({'PIN ID': init_test_data['PIN ID'], 'MAIA-Noticing': noticing, 'MAIA-Not Distracting': not_distracting,
                            'MAIA-Not Worrying': not_worrying, 'MAIA-Attention Regulation': attention_regulation,
                            'MAIA-Emotional Awareness': emotional_awareness, 'MAIA-Self-Regulation': self_regulation,
                            'MAIA-Body Listening': body_listening, 'MAIA-Trusting': trusting, 'MAIA-Awareness': awareness})

#%% Conversion of SRIS-20
'''
Now let's score SRIS-20. The scores are as follows:
SRIS_1 (R), SRIS_8 (R), SRIS_10, SRIS_13 (R), SRIS_16, SRIS_19 are Engagement in Self-reflection Sub-scale.
SRIS_2 (R), SRIS_5, SRIS_7, SRIS_12, SRIS_15, SRIS_18 are Need for Self-reflection Sub-scale.
SRIS_3, SRIS_4 (R), SRIS_6, SRIS_9 (R), SRIS_11 (R), SRIS_14 (R), SRIS_17 (R), SRIS_20 are Insight Sub-scale.
The (R) items will be scored as 7-x, where x is the score for that item.

Let's also pull out items SRIS_12 and SRIS_20 as separate columns titled 'Important to SR' and 'Knowledge abt feelings'.
'''

engagement = (7-init_test_2[['SRIS_1', 'SRIS_8', 'SRIS_13']].mean(axis=1) + init_test_2[['SRIS_10', 'SRIS_16', 'SRIS_19']].mean(axis=1))/2
need = (7-init_test_2[['SRIS_2']].mean(axis=1) + init_test_2[['SRIS_5', 'SRIS_7', 'SRIS_12', 'SRIS_15', 'SRIS_18']].mean(axis=1))/2
insight = (7-init_test_2[['SRIS_4', 'SRIS_9', 'SRIS_11', 'SRIS_14', 'SRIS_17']].mean(axis=1) + init_test_2[['SRIS_3', 'SRIS_6', 'SRIS_20']].mean(axis=1))/2
important = init_test_2['SRIS_12']
knowledge = init_test_2['SRIS_20']

init_test4 = pd.DataFrame({'PIN ID': init_test_data['PIN ID'], 'SRIS-Engagement': engagement, 'SRIS-Need': need,
                            'SRIS-Insight': insight, 'SRIS-Important to SR': important, 'SRIS-Knowledge abt feelings': knowledge})

# %% Interpersonal Reactivity Index (IRI) scoring and conversion
'''
Now let's score IRI. The scores are as follows:
IRI_3 (R), IRI_8, IRI_11, IRI_15 (R), IRI_21, IRI_25, IRI_28 - Perspective Taking Sub-scale.
IRI_1, IRI_5, IRI_7 (R), IRI_12 (R), IRI_16, IRI_23, IRI_26 - Fantasy Sub-scale.
IRI_2, IRI_4 (R), IRI_9, IRI_14 (R), IRI_18 (R), IRI_20, IRI_22 - Empathic Concern Sub-scale.
IRI_6, IRI_10, IRI_13 (R), IRI_17, IRI_19 (R), IRI_24, IRI_27 - Personal Distress Sub-scale.
(R) items need to be subtracted from 5.
'''

perspective = (5-init_test_2[['IRI_3', 'IRI_15']].mean(axis=1) + init_test_2[['IRI_8', 'IRI_11', 'IRI_21', 'IRI_25', 'IRI_28']].mean(axis=1))/2
fantasy = (init_test_2[['IRI_1', 'IRI_5', 'IRI_16', 'IRI_23', 'IRI_26']].mean(axis=1) + 5-init_test_2[['IRI_7', 'IRI_12']].mean(axis=1))/2
empathic = (init_test_2[['IRI_2', 'IRI_9', 'IRI_20', 'IRI_22']].mean(axis=1) + 5-init_test_2[['IRI_4', 'IRI_14', 'IRI_18']].mean(axis=1))/2
distress = (init_test_2[['IRI_6', 'IRI_10', 'IRI_17', 'IRI_24', 'IRI_27']].mean(axis=1) + 5-init_test_2[['IRI_13', 'IRI_19']].mean(axis=1))/2

init_test5 = pd.DataFrame({'PIN ID': init_test_data['PIN ID'], 'IRI-Perspective Taking': perspective, 'IRI-Fantasy': fantasy,
                            'IRI-Empathic Concern': empathic, 'IRI-Personal Distress': distress})
# %% Aggregate into one dataframe

init_test_final = pd.merge(init_test3, init_test4, on='PIN ID')
init_test_final = pd.merge(init_test_final, init_test5, on='PIN ID')

# %% Let's save the final aggregated dataframe as a csv file

init_test_final.to_csv(input+'init_test_cleaned.csv', index=False)

# %%
