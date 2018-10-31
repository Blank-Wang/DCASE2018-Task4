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

def ids_to_multinomial(ids):
    """Ids of wav to multinomial representation. 
    """
    y = np.zeros(len(cfg.lbs))
    for id in ids:
        index = cfg.lb_to_idx[id]
        y[index] = 1
    return y

df1 = pd.read_csv('./metadata/train/train_file_available_weak.csv', sep='\t')

df2 = pd.read_csv('./auto_labels/One-label-0.9.csv', sep='\t')
df2.rename(columns={'fname':'filename', 'label':'event_labels'}, inplace = True)
df2=df2[~df2['event_labels'].isin(['None'])]

df3 = pd.read_csv('./auto_labels/Two-labels-0.49.csv', sep='\t')
df3.rename(columns={'fname':'filename', 'label':'event_labels'}, inplace = True)
df3=df3[~df3['event_labels'].isin(['None'])]

df4 = pd.read_csv('./auto_labels/Three-labels-0.3.csv', sep='\t')
df4.rename(columns={'fname':'filename', 'label':'event_labels'}, inplace = True)
df4=df4[~df4['event_labels'].isin(['None'])]

dfn = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
# dfn = pd.concat([df1, df2], axis=0, ignore_index=True)

dfn.drop_duplicates(inplace = True)

print('Original train files: '+ str(df1.iloc[:,0].size))
# print('Indomain train files: '+ str(df2.iloc[:,0].size))
print('Extended train files: '+ str(dfn.iloc[:,0].size))

dfn.to_csv("./metadata/train/train_file_including_indomain-{0}.csv".format('0.9-0.49-0.3'), index=False, sep='\t', columns=['filename','event_labels'])
# dfn.to_csv("./metadata/train/train_file_including_indomain-{0}.csv".format('0.9999-0-0'), index=False, sep='\t', columns=['filename','event_labels'])
##
y1_all=[]
y2_all=[]
lis=df1.values
for li in lis:
    na=li[0]
    elabels=li[1]
    elabels = elabels.split(',')
    y = ids_to_multinomial(elabels)
    y1_all.append(y)
    
lis=dfn.values
for li in lis:
    na=li[0]
    elabels=li[1]
    elabels = elabels.split(',')
    y = ids_to_multinomial(elabels)
    y2_all.append(y)

y1_all = np.array(y1_all)
y2_all = np.array(y2_all)

cla1=np.sum(y1_all,axis=0)
cla2=np.sum(y2_all,axis=0)

print('train_weak_classes_num:' + str(cla1))
print('train_indomain_classes_num:' + str(cla2))


        