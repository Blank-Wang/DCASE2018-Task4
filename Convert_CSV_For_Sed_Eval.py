'''
Created on Jul 13, 2018

@author: Dezhi Wang
'''
import csv
from itertools import groupby
import pandas as pd
import os
import numpy as np
import config as cfg
import sed_eval
import dcase_util
import shutil

df1 = pd.read_csv('./metadata/test/test_file_available_strong.csv', sep='\t')
df2 = pd.read_csv('./submissions/crnn_sed/sed_submission.csv', sep='\t', header=None)
print(list(df1))
print(list(df2))

df2.columns=list(df1)

df1['scene_label']='None'
df2['scene_label']='None'

df1.rename(columns={'onset':'event onset','offset':'event offset','event_label':'event label'},inplace=True)
df2.rename(columns={'onset':'event onset','offset':'event offset','event_label':'event label'},inplace=True)

df1=df1[['filename','scene_label','event onset','event offset','event label']]
df2=df2[['filename','scene_label','event onset','event offset','event label']]

df1.set_index('filename',inplace=True, drop=False)
df2.set_index('filename',inplace=True, drop=False)

df1.sort_index(inplace=True)
df2.sort_index(inplace=True)

df1.to_csv('./test_ref.txt',index=False, header=False, sep='\t')
df2.to_csv('./test_est.txt',index=False, header=False, sep='\t')

grouped1 = df1.groupby(df1['filename'])
grouped2 = df2.groupby(df2['filename'])

filelist=[]
filelist2=[]

if os.path.exists('./Sed_evaluation/'):
    shutil.rmtree('./Sed_evaluation/')
os.makedirs('./Sed_evaluation/')
for key, group in grouped1:
    filelist.append(key)
    group.to_csv('./Sed_evaluation/'+os.path.splitext(key)[0]+'_ref.txt',index=False, header=False, sep='\t')

for key, group in grouped2:
    filelist2.append(key)
#     if key=='Y4vSnw4-qFao_14.000_24.000.wav':
#         print('targeted')
    if pd.isnull(group.ix[0,'event label']):
        group_tmp=group
        group_tmp.loc[key,'event onset']=3.0
        group_tmp.loc[key,'event offset']=7.0
        group_tmp.loc[key,'event label']='Speech'
        print('none of events can be recognized in '+key)
        group_tmp.to_csv('./Sed_evaluation/'+os.path.splitext(key)[0]+'_est.txt',index=False, header=False, sep='\t')
    else:
        group.to_csv('./Sed_evaluation/'+os.path.splitext(key)[0]+'_est.txt',index=False, header=False, sep='\t')


print('file_number_of_testset:',len(filelist))
print('file_number_of_estset:',len(filelist2))


data = []
# Get used event labels
all_data = dcase_util.containers.MetaDataContainer()
for file_name in filelist:
    reference_event_list = sed_eval.io.load_event_list(
        filename='./Sed_evaluation/'+os.path.splitext(file_name)[0]+'_ref.txt'
    )
    estimated_event_list = sed_eval.io.load_event_list(
        filename='./Sed_evaluation/'+os.path.splitext(file_name)[0]+'_est.txt'
    )

    data.append({'reference_event_list': reference_event_list,
                 'estimated_event_list': estimated_event_list})

    all_data += reference_event_list

event_labels = all_data.unique_event_labels

#Create metrics classes, define parameters
segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
    event_label_list=event_labels,
    time_resolution=1.0
)

event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
    event_label_list=event_labels,
    t_collar=0.2,
    percentage_of_length=0.2
)

# Go through files
for file_pair in data:
    segment_based_metrics.evaluate(
        reference_event_list=file_pair['reference_event_list'],
        estimated_event_list=file_pair['estimated_event_list']
    )

    event_based_metrics.evaluate(
        reference_event_list=file_pair['reference_event_list'],
        estimated_event_list=file_pair['estimated_event_list']
    )

# # Get only certain metrics
overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
print("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])
# 
# # Or print all metrics as reports
print(segment_based_metrics)
print(event_based_metrics)




        