'''
Created on Jul 13, 2018

@author: Dezhi Wang
'''
import csv
from itertools import groupby
import pandas as pd
import os
import numpy as np

df = pd.read_csv('./metadata/test/test.csv', sep='\t')
wav_dir= "/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/test"
names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))
cnt=0
innt=0
for index,row in df.iterrows():
    fe_path = os.path.join(wav_dir, row['filename'])
    if not os.path.isfile(fe_path):  
        print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
        df.drop(index, axis=0, inplace=True)
        cnt +=1
cnt1=0
cnt2=0
cnt3=0  
cnt0=0
cnt4=0
grouped = df['event_label'].groupby(df['filename'])
for key, group in grouped:
    innt+=1
    rows = list(group)
    rows= [rows]
    columns = zip(*(r for r in rows))
    use_values = list(set(columns))
    if isinstance(use_values[0][0],str):
        if len(use_values)==1:
            cnt1+=1
        elif len(use_values)==2:
            cnt2+=1
        elif len(use_values)==3:
            cnt3+=1
        elif len(use_values)==4:
            cnt4+=1
    else:
        cnt0+=1
print('The total number of missing files is ' +str(cnt)+' out of '+str(innt))    
print('Numbers of types of events in test dataset are: one-type-{0}, two-types-{1}, three-types-{2}'.format(str(cnt1),str(cnt2),str(cnt3))) 
print('0-Type--'+str(cnt0)+'-----4-Type--'+str(cnt4)+'-----all-number--'+str(cnt0+cnt1+cnt2+cnt3+cnt4))  
print('test_csv_files_finished')

################################################################################################################

df = pd.read_csv('./metadata/train/weak.csv', sep='\t')
wav_dir= "/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/train/weak"
names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))
cnt=0
innt=0
for index,row in df.iterrows():
    fe_path = os.path.join(wav_dir, row['filename'])
    if not os.path.isfile(fe_path):  
        print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
        df.drop(index, axis=0, inplace=True)
        cnt +=1
cnt1=0
cnt2=0
cnt3=0  
cnt0=0
cnt4=0
grouped = df['event_labels'].groupby(df['filename'])
for key, group in grouped:
    innt+=1
    rows = list(group)
    use_values = rows[0].split(',')    
    if isinstance(use_values[0],str):
        if len(use_values)==1:
            cnt1+=1
        elif len(use_values)==2:
            cnt2+=1
        elif len(use_values)==3:
            cnt3+=1
        elif len(use_values)==4:
            cnt4+=1
    else:
        cnt0+=1
print('The total number of missing files is ' +str(cnt)+' out of '+str(innt))    
print('Numbers of types of events in train weak dataset are: one-type-{0}, two-types-{1}, three-types-{2}'.format(str(cnt1),str(cnt2),str(cnt3))) 
print('0-Type--'+str(cnt0)+'-----4-Type--'+str(cnt4)+'-----all-number--'+str(cnt0+cnt1+cnt2+cnt3+cnt4))  
print('train_weak_csv_files_finished')

############################################################################################

df = pd.read_csv('./metadata/eval/eval_ground_truth.csv', sep='\t')
wav_dir= "/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/eval"
names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))
cnt=0
innt=0
for index,row in df.iterrows():
    fe_path = os.path.join(wav_dir, row['filename'])
    if not os.path.isfile(fe_path):  
        print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
        df.drop(index, axis=0, inplace=True)
        cnt +=1
cnt1=0
cnt2=0
cnt3=0  
cnt0=0
cnt4=0
grouped = df['event_label'].groupby(df['filename'])
for key, group in grouped:
    innt+=1
    rows = list(group)
    rows= [rows]
    columns = zip(*(r for r in rows))
    use_values = list(set(columns))
    if isinstance(use_values[0][0],str):
        if len(use_values)==1:
            cnt1+=1
        elif len(use_values)==2:
            cnt2+=1
        elif len(use_values)==3:
            cnt3+=1
        elif len(use_values)==4:
            cnt4+=1
    else:
        cnt0+=1

print('The total number of missing files is ' +str(cnt)+' out of '+str(innt))    
print('Numbers of types of events in eval dataset are: one-type-{0}, two-types-{1}, three-types-{2}'.format(str(cnt1),str(cnt2),str(cnt3))) 
print('0-Type--'+str(cnt0)+'-----4-Type--'+str(cnt4)+'-----all-number--'+str(cnt0+cnt1+cnt2+cnt3+cnt4))  
print('eval_csv_files_finished')


df = pd.read_csv('./metadata/train/train_file_including_indomain-0.9-0.35-0.15.csv', sep='\t')
wav_dir= "/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/train/weak_and_indomain"
names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))
cnt=0
innt=0
for index,row in df.iterrows():
    fe_path = os.path.join(wav_dir, row['filename'])
    if not os.path.isfile(fe_path):  
        print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
        df.drop(index, axis=0, inplace=True)
        cnt +=1
cnt1=0
cnt2=0
cnt3=0  
cnt0=0
cnt4=0
grouped = df['event_labels'].groupby(df['filename'])
for key, group in grouped:
    innt+=1
    rows = list(group)
    use_values = rows[0].split(',')    
    if isinstance(use_values[0],str):
        if len(use_values)==1:
            cnt1+=1
        elif len(use_values)==2:
            cnt2+=1
        elif len(use_values)==3:
            cnt3+=1
        elif len(use_values)==4:
            cnt4+=1
    else:
        cnt0+=1
print('The total number of missing files is ' +str(cnt)+' out of '+str(innt))    
print('Numbers of types of events in train weak dataset are: one-type-{0}, two-types-{1}, three-types-{2}'.format(str(cnt1),str(cnt2),str(cnt3))) 
print('0-Type--'+str(cnt0)+'-----4-Type--'+str(cnt4)+'-----all-number--'+str(cnt0+cnt1+cnt2+cnt3+cnt4))  
print('train_weak_csv_files_finished')

        
        
        