#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 01:12:03 2022

@author: ksalibay
This script generates a Sankey diagram for the lobar structure parcellation data
"""
import pandas as pd
import plotly.express as px
import kaleido
#%% Declare vars and import data
subject = 's5'
subject2 = 'S5'
band = 'gamma'
df = pd.read_csv(
    '/N/slate/ksalibay/DataICM2019/'+subject+'_conmats/'+subject+'_diastole_'+band+'_sankey_fin_num.txt', sep=',',
    names=['Label','Diastole','Systole','Lobe']
)
##%%

# dia_dim = go.parcats.Dimension(values=df.Diastole, label="Diastole")

# sys_dim = go.parcats.Dimension(values=df.Systole, label="Systole")
# color = df.Lobe;
# colorscale = 'rainbow';
# colorbar = {'tickvals': ['Frontal', 'Insular', 'Limbic', 'Temporal', 'Parietal', 'Occipital'], 
#             'ticktext': ['Frontal', 'Insular', 'Limbic', 'Temporal', 'Parietal', 'Occipital']};
# fig = go.Figure(data = [go.Parcats(dimensions=[dia_dim,sys_dim],
#         line={'color': color, 'colorscale': colorscale},
#         hoveron='color', hoverinfo='skip',
#         labelfont={'size': 18, 'family': 'Times'},
#         tickfont={'size': 16, 'family': 'Times'},
#         arrangement='freeform')])
# fig = px.parallel_categories(df, dimensions=['Diastole','Systole'], color='Lobe', 
#                             color_continuous_scale=px.colors.sequential.Inferno)
# #fig.show()
# fig.write_image(file='s2_sankey_'+band+'_lobar.png')
#%% Ordinal
#Find the ordinal number of the community that is >1 in both sys and dia

counts_sys = df['Systole'].value_counts(ascending = True).loc[lambda x : x==1].to_frame().sort_index()
counts_dia = df['Diastole'].value_counts(ascending = True).loc[lambda x : x==1].to_frame().sort_index()

sys_smallcomm = counts_sys.idxmin()[0]
dia_smallcomm = counts_dia.idxmin()[0]

##%% Generate first figure (staying)

# df1 = df.loc[(df['Diastole'] == df['Systole']) | ((df['Diastole'] == 1) & (df['Systole'] == 2))
#              | ((df['Diastole'] == 2) & (df['Systole'] == 1)) #| (df['Diastole'] == 3) 
#              | ((df['Diastole'] >= dia_smallcomm) & (df['Systole'] >= sys_smallcomm))]
# fig1 = px.parallel_categories(df1, dimensions=['Diastole','Systole'], color='Lobe', 
#                             color_continuous_scale=[(0.00, "rgb(0,0,4)"),   (0.16, "rgb(0,0,4)"),
#                                                      (0.16, "rgb(74,12,107)"), (0.33, "rgb(74,12,107)"),
#                                                      (0.33, "rgb(165,44,96)"),  (0.5, "rgb(165,44,96)"),
#                                                      (0.5,"rgb(207,84,70)"),(0.66,"rgb(207,84,70)"),
#                                                      (0.66,"rgb(251,155,6)"),(0.83,"rgb(251,155,6)"),
#                                                      (0.83,"rgb(220,209,164)"),(1,"rgb(220,209,164)")])
# fig1.show()
# fig1.write_image(file='/N/slate/ksalibay/DataICM2019/'+subject+'_data/'+subject+'_sankey_'+band+'_only_stay.png')

# df1.to_csv('/N/slate/ksalibay/DataICM2019/'+subject+'_data/'+subject+'_comms_'+band+'_only_stay.csv')

##%% Generate second figure (moving)

# df2 = df.loc[set(df.index) - set(df1.index)]
# fig2 = px.parallel_categories(df2, dimensions=['Diastole','Systole'], color='Lobe', 
#                             color_continuous_scale=[(0.00, "rgb(0,0,4)"),   (0.16, "rgb(0,0,4)"),
#                                                      (0.16, "rgb(74,12,107)"), (0.33, "rgb(74,12,107)"),
#                                                      (0.33, "rgb(165,44,96)"),  (0.5, "rgb(165,44,96)"),
#                                                      (0.5,"rgb(207,84,70)"),(0.66,"rgb(207,84,70)"),
#                                                      (0.66,"rgb(251,155,6)"),(0.83,"rgb(251,155,6)"),
#                                                      (0.83,"rgb(220,209,164)"),(1,"rgb(220,209,164)")])
# fig2.write_image(file='/N/slate/ksalibay/DataICM2019/'+subject+'_data/'+subject+'_sankey_'+band+'_only_move.png')

# df2.to_csv('/N/slate/ksalibay/DataICM2019/'+subject+'_data/'+subject+'_comms_'+band+'_only_move.csv')

#%% Generate stay figure PlotlyGO
#Playing around with Plotly Parallel categories plots to generate prettier figures here
import plotly.graph_objects as go
df1 = df.loc[(df['Diastole'] == df['Systole']) | ((df['Diastole'] == 1) & (df['Systole'] == 2))
             | ((df['Diastole'] == 2) & (df['Systole'] == 1)) | (df['Diastole'] == 3) | (df['Systole'] == 3)
             | ((df['Diastole'] >= dia_smallcomm) & (df['Systole'] >= sys_smallcomm))]
categorical_dimensions = ['Systole', 'Diastole'];

dimensions = [dict(values=df1[label], label=label) for label in categorical_dimensions]

color = df1.Lobe;

fig3 = go.Figure(go.Parcats(
    dimensions=dimensions,
    line={'color': color, 'cmin':1, 'cmax':6,
          'colorscale': [(0.00, "rgb(0,0,4)"),   (0.16, "rgb(0,0,4)"),
                                                     (0.16, "rgb(74,12,107)"), (0.33, "rgb(74,12,107)"),
                                                     (0.33, "rgb(165,44,96)"),  (0.5, "rgb(165,44,96)"),
                                                     (0.5,"rgb(207,84,70)"),(0.66,"rgb(207,84,70)"),
                                                     (0.66,"rgb(251,155,6)"),(0.83,"rgb(251,155,6)"),
                                                     (0.83,"rgb(220,209,164)"),(1,"rgb(220,209,164)")], 
          'shape': 'hspline', 
          'colorbar':{'tick0':1, 'dtick':1, 'tickvals':list(range(1,7)),
                      'ticktext':['Frontal','Insular','Limbic','Temporal','Parietal','Occipital'],
                      'orientation':'v','ticks':'outside','ticklen':10}},
))
fig3.update_layout(showlegend=True, legend_title_text='Lobe')
fig3.update_layout(title_text="Labels staying within original communities, "+band+" band", 
                   title_x = 0.5,
                   title_xanchor="center", title_xref = "paper")
fig3.write_image(file='/N/slate/ksalibay/DataICM2019/'+subject2+'_data/'+subject+'_sankey_'+band+'_only_stay_plotlygo.png')
df1.to_csv('/N/slate/ksalibay/DataICM2019/'+subject2+'_data/'+subject+'_comms_'+band+'_only_stay.csv')
#%% Generate move figure PlotlyGO

df2 = df.loc[list(set(df.index) - set(df1.index))]
categorical_dimensions = ['Systole', 'Diastole'];

dimensions = [dict(values=df2[label], label=label) for label in categorical_dimensions]

color = df2.Lobe;

fig4 = go.Figure(go.Parcats(
    dimensions=dimensions,
    line={'color': color, 'cmin':1, 'cmax':6,
          'colorscale': [(0.00, "rgb(0,0,4)"),   (0.16, "rgb(0,0,4)"),
                                                     (0.16, "rgb(74,12,107)"), (0.33, "rgb(74,12,107)"),
                                                     (0.33, "rgb(165,44,96)"),  (0.5, "rgb(165,44,96)"),
                                                     (0.5,"rgb(207,84,70)"),(0.66,"rgb(207,84,70)"),
                                                     (0.66,"rgb(251,155,6)"),(0.83,"rgb(251,155,6)"),
                                                     (0.83,"rgb(220,209,164)"),(1,"rgb(220,209,164)")], 
          'shape': 'hspline', 
          'colorbar':{'tick0':1, 'dtick':1, 'tickvals':list(range(1,7)),
                      'ticktext':['Frontal','Insular','Limbic','Temporal','Parietal','Occipital'],
                      'orientation':'v','ticks':'outside','ticklen':10}},
))
fig4.update_layout(showlegend=True, legend_title_text='Lobe')
fig4.update_layout(title_text="Labels moving to a different community, "+band+" band", 
                   title_x = 0.5,
                   title_xanchor="center", title_xref = "paper")
fig4.write_image(file='/N/slate/ksalibay/DataICM2019/'+subject2+'_data/'+subject+'_sankey_'+band+'_only_move_plotlygo.png')
df2.to_csv('/N/slate/ksalibay/DataICM2019/'+subject2+'_data/'+subject+'_comms_'+band+'_only_move.csv')
