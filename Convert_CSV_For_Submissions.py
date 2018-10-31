#!/usr/bin/env python
# -*- coding: utf-8 -*- 
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

# df1 = pd.read_csv('./metadata/eval/eval_file_available_strong_labels.csv', sep='\t')
df1 = pd.read_csv('./metadata/eval/eval.csv', sep='\t')
df2 = pd.read_csv('./submissions/combined-20.73/sed_submission_eval.csv', sep='\t', header=None)
print(list(df1))
print(list(df2))
print(df1.iloc[:,0].size)
print(df2.iloc[:,0].size)
df2.columns=list(['filename','event onset','event offset','event label'])
# df1['scene_label']='None'
# df2['scene_label']='None'
# df1.rename(columns={'onset':'event onset','offset':'event offset','event_label':'event label'},inplace=True)
# df2.rename(columns={'onset':'event onset','offset':'event offset','event_label':'event label'},inplace=True)
# df1=df1[['filename','scene_label','event onset','event offset','event label']]
# df2=df2[['filename','scene_label','event onset','event offset','event label']]

df1.set_index('filename',inplace=True, drop=False)
df2.set_index('filename',inplace=True, drop=False)

df1.sort_index(inplace=True)
df2.sort_index(inplace=True)

outliers=list(set(df1.index.values)-set(df2.index.values))
print('Check these files: '+str(outliers))

df_missing = pd.DataFrame(outliers, columns=['filename']) 
df_missing.to_csv('./evaluation_missing_files.csv',index=False, sep='\t')

# df1.drop(index= outliers[:],inplace=True)
# df1.reset_index(drop=True)

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




        