'''
Created on Jul 13, 2018

@author: Dezhi Wang
'''
import csv
from itertools import groupby
import pandas as pd
import os


# TEST_WAV_DIR="/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/test"
# TRAIN_WAV_DIR="/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/train/weak"
# TRAIN_WAV_DIR_InDomain="/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/train/unlabel_in_domain"
# TRAIN_WAV_DIR_OutOfDomain="/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/train/unlabel_out_of_domain"
# EVALUATION_WAV_DIR="/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/eval"

df = pd.read_csv('./metadata/test/test.csv', sep='\t')
new_key=[]
new_event=[]
wav_dir= "/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/test"

names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in df.iterrows():
        
        fe_path = os.path.join(wav_dir, row['filename'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
            df.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
df.to_csv("./metadata/test/test_file_available_strong.csv", index=False, sep='\t', columns=['filename','onset','offset','event_label'])

  
grouped = df['event_label'].groupby(df['filename'])
for key, group in grouped:
    rows = list(group)
    rows= [rows]
#         rows = [rows[0], rows[-1]]
#         columns = zip(*(r[0:] for r in rows))
    columns = zip(*(r for r in rows))
#         use_values = [max(c) for c in columns]
    use_values = list(set(columns))
#     new_row = [key] + [r[0] for r in use_values]
    list_str = ','.join([r[0] for r in use_values])
    new_key.append(key)
#     new_event.append([r[0] for r in use_values])
    new_event.append(list_str)

df=pd.DataFrame({'event_label':new_event, 'filename':new_key})
df.to_csv("./metadata/test/test_file_available_weak.csv", index=False, sep='\t', columns=['filename','event_label'])

print('test_csv_files_finished')

# remove the unavailable files in the train directory
df = pd.read_csv('./metadata/train/weak.csv', sep='\t')
wav_dir= "/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/train/weak"

names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in df.iterrows():
        
        fe_path = os.path.join(wav_dir, row['filename'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
            df.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
df.to_csv("./metadata/train/train_file_available_weak.csv", index=False, sep='\t')
# df.to_csv("./metadata/train/train_file_available_weak.csv", index=False, sep=',', columns=['filename','event_label'])

print('train_weak_csv_files_finished')

####in domain
df = pd.read_csv('./metadata/train/unlabel_in_domain.csv', sep='\t')
wav_dir= "/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/train/unlabel_in_domain"

names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in df.iterrows():
        
        fe_path = os.path.join(wav_dir, row['filename'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
            df.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
df.to_csv("./metadata/train/train_file_available_unlabel_in_domain.csv", index=False, sep='\t')
# df.to_csv("./metadata/train/train_file_available_weak.csv", index=False, sep=',', columns=['filename','event_label'])

print('train_unlabel_in_domain_csv_files_finished')


### out of domain
df = pd.read_csv('./metadata/train/unlabel_out_of_domain.csv', sep='\t')
wav_dir= "/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/train/unlabel_out_of_domain"

names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in df.iterrows():
        
        fe_path = os.path.join(wav_dir, row['filename'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
            df.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
df.to_csv("./metadata/train/train_file_available_unlabel_out_of_domain.csv", index=False, sep='\t')
# df.to_csv("./metadata/train/train_file_available_weak.csv", index=False, sep=',', columns=['filename','event_label'])

print('train_unlabel_out_of_domain_csv_files_finished')



#### eval

df = pd.read_csv('./metadata/eval/eval.csv', sep='\t')
wav_dir= "/media/user/2b62472c-494b-4e74-9409-498d5428db622/DCASE2018-Dataset-Task4/audio/eval"

names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
names = sorted(names)
print("Total file number: %d" % len(names))

cnt=0
for index,row in df.iterrows():
        
        fe_path = os.path.join(wav_dir, row['filename'])
        
        if not os.path.isfile(fe_path):  
            print("File %s is in the csv file but the wav file does not exist!" % row['filename'])     
            df.drop(index, axis=0, inplace=True)
            cnt +=1
print('The total number of missing files is %d' %cnt)
df.to_csv("./metadata/eval/eval_file_available.csv", sep='\t', index=False)
# df.to_csv("./metadata/train/train_file_available_weak.csv", index=False, sep=',', columns=['filename','event_label'])

print('eval_csv_files_finished')


        
        
        