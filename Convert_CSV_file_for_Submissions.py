#!/usr/bin/env python
# -*- coding: utf-8 -*- 
'''
Created on Jul 13, 2018

@author: user
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

path='/media/user/Honor/DCASE2018-Task4-提交/task4-Dezhi Wang-NUDT/CSV-files/crnn_sed-new-21.63'
# path='/media/user/Honor/DCASE2018-Task4-提交/task4-Dezhi Wang-NUDT/CSV-files/crnn_sed-new-21.56'
# path='/media/user/Honor/DCASE2018-Task4-提交/task4-Dezhi Wang-NUDT/CSV-files/crnn_sed-new-19.97'
# path='/media/user/Honor/DCASE2018-Task4-提交/task4-Dezhi Wang-NUDT/CSV-files/crnn_sed-new-19.70'

# df1 = pd.read_csv('./metadata/eval/eval_file_available_strong_labels.csv', sep='\t')
df1 = pd.read_csv('./metadata/eval/eval.csv', sep='\t')
df2 = pd.read_csv(path+'/sed_submission_eval.csv', sep='\t', header=None)
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
# df_missing = pd.DataFrame(outliers, columns=['filename']) 
# df_missing.to_csv('./Task4-evaluation_missing_files.csv',index=False, sep='\t')

# dfn = pd.concat([df2, df_missing], axis=0, ignore_index=True)
# 
# print('Original train files: '+ str(df2.iloc[:,0].size))
# print('Missing train files: '+ str(df_missing.iloc[:,0].size))
# print('Extended train files: '+ str(dfn.iloc[:,0].size))

# dfn.set_index('filename',inplace=True, drop=False)
df2.to_csv(path+'/sed_submission_eval_processed.csv', index=False, sep='\t', columns=['filename','event onset','event offset','event label'])

# outliers_check=list(set(df1.index.values)-set(dfn.index.values))
# print('Check these files: '+str(outliers_check))



        