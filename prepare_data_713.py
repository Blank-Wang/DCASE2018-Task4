'''
Created on 2018-5-8

@author: Dezhi Wang
'''
from __future__ import print_function
import numpy as np
import sys
import soundfile
import os
import librosa
from scipy import signal
# import pickle
import cPickle
import scipy
import time
import csv
import gzip
import h5py
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import argparse
import pandas as pd
import config as cfg

# Read wav
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    L = len(audio)
    max_len=int(10*target_fs)
    if L < max_len:
        pad_shape = max_len - L 
        pad = np.zeros(pad_shape)
        audio= np.concatenate((audio, pad), axis=0)
    else:
        audio = audio[0:max_len]
    
    return audio, fs
    
# Write wav
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

# Create an empty folder
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        

### Feature extraction. 
def extract_features(wav_dir, out_dir, recompute=True):
    """Extract log mel spectrogram features. 
    
    Args:
      wav_dir: string, directory of wavs. 
      out_dir: string, directory to write out features. 
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features. 
                 
    Returns:
      None
    """
    fs = cfg.sample_rate
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    
    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("Total file number: %d" % len(names))

    # Mel filter bank
    melW = librosa.filters.mel(sr=fs, 
                               n_fft=n_window, 
                               n_mels=64, 
                               fmin=0., 
                               fmax=fs/2.)
    
    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + os.path.splitext(na)[0] + '.p'
        
        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)
            (audio, _) = read_audio(wav_path, fs)
            
            # Skip corrupted wavs
            if audio.shape[0] == 0:
                print("File %s is corrupted!" % wav_path)
            else:
                # Compute spectrogram
                ham_win = np.hamming(n_window)
                [f, t, x] = signal.spectral.spectrogram(
                                x=audio, 
                                window=ham_win,
                                nperseg=n_window, 
                                noverlap=n_overlap, 
                                detrend=False, 
                                return_onesided=True, 
                                mode='magnitude') 
                x = x.T
                x = np.dot(x, melW.T)
                x = np.log(x + 1e-8)
                x = x.astype(np.float32)
                
                x_delta1 = librosa.feature.delta(x, order=1)
                x_delta2 = librosa.feature.delta(x, order=2)
                
                x_mix=[]
                x_mix.append(x[0:480,:])
                x_mix.append(x_delta1[0:480,:])
                x_mix.append(x_delta2[0:480,:])
                x_mix=np.array(x_mix)
                x_mix=np.transpose(x_mix, (1,2,0))
                # Dump to pickle
                cPickle.dump(x_mix, open(out_path, 'wb'),   
                             protocol=cPickle.HIGHEST_PROTOCOL)
        cnt += 1
    print("Extracting feature time: %s" % (time.time() - t1,))

### Pack features of hdf5 file
def pack_features_to_hdf5_train(fe_dir, csv_path, out_path):
    """Pack extracted features to a single hdf5 file. 
    
    This hdf5 file can speed up loading the features. This hdf5 file has 
    structure:
       na_list: list of names
       x: bool array, (n_clips)
       y: float32 array, (n_clips, n_time, n_freq)
       
    Args: 
      fe_dir: string, directory of features. 
      csv_path: string | "", path of csv file. E.g. "testing_set.csv". If the 
          string is empty, then pack features with all labels False. 
      out_path: string, path to write out the created hdf5 file. 
      
    Returns:
      None
    """
    max_len = cfg.max_len
    create_folder(os.path.dirname(out_path))
    
    t1 = time.time()
    x_all, y_all, na_all = [], [], []
    
    if csv_path != "":    # Pack from csv file (training & testing from dev. data)
#         with open(csv_path, 'rb') as f:
#             reader = csv.reader(f)
#             lis = list(reader)
#         llist=pd.read_csv(csv_path, header=None)
#         lis=llist.values
        llist=pd.read_csv(csv_path, sep='\t')
        lis=llist.values
#         lis=list(pd.read_csv(csv_path, sep='\t'))
        cnt = 0
        for li in lis:
            na=li[0]
            elabels=li[1]
            if cnt % 100 == 0: print(cnt)
#             na = os.path.splitext(na)[0]
#             bare_na = 'Y' + na + '_' + bgn + '_' + fin # Correspond to the wav name. 
            bare_na=os.path.splitext(na)[0]
            fe_na = bare_na + ".p"
            fe_path = os.path.join(fe_dir, fe_na)
            
            if not os.path.isfile(fe_path):
                print("File %s is in the csv file but the feature is not extracted!" % fe_path)
            else:
                na_all.append(bare_na[0:] + ".wav") # Remove 'Y' in the begining. 
                x = cPickle.load(open(fe_path, 'rb'))
                x = pad_trunc_seq(x, max_len) #fixed length
                x_all.append(x)
#                 ids = ids.split(',')
#                 y = ids_to_multinomial(ids)
                elabels = elabels.split(',')
                y = ids_to_multinomial(elabels)
                y_all.append(y)
            cnt += 1
    else:   # Pack from features without ground truth label (dev. data)
        names = os.listdir(fe_dir)
        names = sorted(names)
        for fe_na in names:
            bare_na = os.path.splitext(fe_na)[0]
            fe_path = os.path.join(fe_dir, fe_na)
            na_all.append(bare_na + ".wav")
            x = cPickle.load(open(fe_path, 'rb'))
            x = pad_trunc_seq(x, max_len)
            x_all.append(x)
            y_all.append(None)
        
    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.bool)
    print("len(na_all): %d", len(na_all))
    print("x_all.shape: %s, %s" % (x_all.shape, x_all.dtype))
    print("y_all.shape: %s, %s" % (y_all.shape, y_all.dtype))
    
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('na_list', data=na_all)
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        
    print("Save hdf5 to %s" % out_path)
    print("Pack features time: %s" % (time.time() - t1,))
 
def pack_features_to_hdf5_test(fe_dir, csv_path, out_path):
    """Pack extracted features to a single hdf5 file. 
    
    This hdf5 file can speed up loading the features. This hdf5 file has 
    structure:
       na_list: list of names
       x: bool array, (n_clips)
       y: float32 array, (n_clips, n_time, n_freq)
       
    Args: 
      fe_dir: string, directory of features. 
      csv_path: string | "", path of csv file. E.g. "testing_set.csv". If the 
          string is empty, then pack features with all labels False. 
      out_path: string, path to write out the created hdf5 file. 
      
    Returns:
      None
    """
    max_len = cfg.max_len
    create_folder(os.path.dirname(out_path))
    
    t1 = time.time()
    x_all, y_all, na_all = [], [], []
    
    if csv_path != "":    # Pack from csv file (training & testing from dev. data)
#         with open(csv_path, 'rb') as f:
#             reader = csv.reader(f)
#             lis = list(reader)
        llist=pd.read_csv(csv_path, sep='\t')
        lis=llist.values
        cnt = 0
        for li in lis:    
#             [na, elabels] = li[0].split()
            na=li[0]
            elabels=li[1]
            if cnt % 100 == 0: print(cnt)
            bare_na = os.path.splitext(na)[0]
#             bare_na = 'Y' + na + '_' + bgn + '_' + fin # Correspond to the wav name. 
            fe_na = bare_na + ".p"
            fe_path = os.path.join(fe_dir, fe_na)
            
            if not os.path.isfile(fe_path):
                print("File %s is in the csv file but the feature is not extracted!" % fe_path)                
            else:
                na_all.append(bare_na[0:] + ".wav") # Remove 'Y' in the begining. 
                x = cPickle.load(open(fe_path, 'rb'))
                x = pad_trunc_seq(x, max_len) #fixed length
                x_all.append(x)
#                 ids = ids.split(',')
#                 y = ids_to_multinomial(ids)
#                 lbs = lbs.split(',')
                elabels = elabels.split(',')
                y = ids_to_multinomial(elabels)
                y_all.append(y)
            cnt += 1
    else:   # Pack from features without ground truth label (dev. data)
        names = os.listdir(fe_dir)
        names = sorted(names)
        for fe_na in names:
            bare_na = os.path.splitext(fe_na)[0]
            fe_path = os.path.join(fe_dir, fe_na)
            na_all.append(bare_na + ".wav")
            x = cPickle.load(open(fe_path, 'rb'))
            x = pad_trunc_seq(x, max_len)
            x_all.append(x)
            y_all.append(None)
        
    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.bool)
    print("len(na_all): %d", len(na_all))
    print("x_all.shape: %s, %s" % (x_all.shape, x_all.dtype))
    print("y_all.shape: %s, %s" % (y_all.shape, y_all.dtype))
    
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('na_list', data=na_all)
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        
    print("Save hdf5 to %s" % out_path)
    print("Pack features time: %s" % (time.time() - t1,))

def ids_to_multinomial(ids):
    """Ids of wav to multinomial representation. 
    
    Args:
      ids: list of id, e.g. ['/m/0284vy3', '/m/02mfyn']
      
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    y = np.zeros(len(cfg.lbs))
    for id in ids:
        index = cfg.lb_to_idx[id]
        y[index] = 1
    return y
    
def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]
    return x_new
    
### Load data & scale data
def load_hdf5_data(hdf5_path, verbose=1):
    """Load hdf5 data. 
    
    Args:
      hdf5_path: string, path of hdf5 file. 
      verbose: integar, print flag. 
      
    Returns:
      x: ndarray (np.float32), shape: (n_clips, n_time, n_freq)
      y: ndarray (np.bool), shape: (n_clips, n_classes)
      na_list: list, containing wav names. 
    """
    t1 = time.time()
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x'))  #(51172,240,64)
        y = np.array(hf.get('y'))
        na_list = list(hf.get('na_list')) #(51172,17) 
        
    if verbose == 1:
        print("--- %s ---" % hdf5_path)
        print("x.shape: %s %s" % (x.shape, x.dtype))
        print("y.shape: %s %s" % (y.shape, y.dtype))
        print("len(na_list): %d" % len(na_list))
        print("Loading time: %s" % (time.time() - t1,))
        
    return x, y, na_list

def calculate_scaler(hdf5_path, out_path):
    """Calculate scaler of input data on each frequency bin. 
    
    Args:
      hdf5_path: string, path of packed hdf5 features file. 
      out_path: string, path to write out the calculated scaler. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    t1 = time.time()
    (x, y, na_list) = load_hdf5_data(hdf5_path, verbose=1)
    (n_clips, n_time, n_freq) = x.shape
    x2d = x.reshape((n_clips * n_time, n_freq))
    scaler = preprocessing.StandardScaler().fit(x2d)
    print("Mean: %s" % (scaler.mean_,))
    print("Std: %s" % (scaler.scale_,))
    print("Calculating scaler time: %s" % (time.time() - t1,))
    cPickle.dump(scaler, open(out_path, 'wb'))
    
def do_scale(x3d, scaler_path, verbose=1):
    """Do scale on the input sequence data. 
    
    Args:
      x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
      scaler_path: string, path of pre-calculated scaler. 
      verbose: integar, print flag. 
      
    Returns:
      Scaled input sequence data. 
    """
    t1 = time.time()
#     scaler = pickle.load(open(scaler_path, 'rb'),encoding='bytes')
    scaler = cPickle.load(open(scaler_path, 'rb'))
    (n_clips, n_time, n_freq) = x3d.shape
    x2d = x3d.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.fit_transform(x2d)
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
    if verbose == 1:
        print("Scaling time: %s" % (time.time() - t1,))
    return x3d_scaled


def calculate_scaler_3CH(hdf5_path, out_path):
    """Calculate scaler of input data on each frequency bin. 
    
    Args:
      hdf5_path: string, path of packed hdf5 features file. 
      out_path: string, path to write out the calculated scaler. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    t1 = time.time()
    (x, y, na_list) = load_hdf5_data(hdf5_path, verbose=1)
    xa=x[:,:,:,0]
    xb=x[:,:,:,1]
    xc=x[:,:,:,2]
    (n_clips, n_time, n_freq) = xa.shape
    xa2d = xa.reshape((n_clips * n_time, n_freq))
    scaler_a = preprocessing.StandardScaler().fit(xa2d)
    xb2d = xb.reshape((n_clips * n_time, n_freq))
    scaler_b = preprocessing.StandardScaler().fit(xb2d)
    xc2d = xc.reshape((n_clips * n_time, n_freq))
    scaler_c = preprocessing.StandardScaler().fit(xc2d)
    print("Mean_CH1: %s" % (scaler_a.mean_,))
    print("Std_CH1: %s" % (scaler_a.scale_,))
    print("Mean_CH2: %s" % (scaler_b.mean_,))
    print("Std_CH2: %s" % (scaler_b.scale_,))
    print("Mean_CH3: %s" % (scaler_c.mean_,))
    print("Std_CH3: %s" % (scaler_c.scale_,))
    print("Calculating scaler time: %s" % (time.time() - t1,))
#     scaler=[scaler_a,scaler_b,scaler_c]
    cPickle.dump([scaler_a,scaler_b,scaler_c], open(out_path, 'wb'))
    
def do_scale_3CH(x3d, scaler_path, verbose=1):
    """Do scale on the input sequence data. 
    
    Args:
      x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
      scaler_path: string, path of pre-calculated scaler. 
      verbose: integar, print flag. 
      
    Returns:
      Scaled input sequence data. 
    """
    t1 = time.time()
    scaler_a,scaler_b,scaler_c = cPickle.load(open(scaler_path, 'rb')) 
#     scaler_a=scaler[0]
#     scaler_b=scaler[1]
#     scaler_c=scaler[2]
    xa=x3d[:,:,:,0]
    xb=x3d[:,:,:,1]
    xc=x3d[:,:,:,2]
    (n_clips, n_time, n_freq) = xa.shape
    xa2d = xa.reshape((n_clips * n_time, n_freq))
    xa_scaled = scaler_a.transform(xa2d)
    xa3d_scaled = xa_scaled.reshape((n_clips, n_time, n_freq))
    xb2d = xb.reshape((n_clips * n_time, n_freq))
    xb_scaled = scaler_b.transform(xb2d)
    xb3d_scaled = xb_scaled.reshape((n_clips, n_time, n_freq))
    xc2d = xc.reshape((n_clips * n_time, n_freq))
    xc_scaled = scaler_c.transform(xc2d)
    xc3d_scaled = xc_scaled.reshape((n_clips, n_time, n_freq))
    x3d[:,:,:,0]=xa3d_scaled
    x3d[:,:,:,1]=xb3d_scaled
    x3d[:,:,:,2]=xc3d_scaled   

    if verbose == 1:
        print("Scaling time: %s" % (time.time() - t1,))
    return x3d

### Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_ef = subparsers.add_parser('extract_features')
    parser_ef.add_argument('--wav_dir', type=str)
    parser_ef.add_argument('--out_dir', type=str)
    parser_ef.add_argument('--recompute', type=bool)
    
    parser_pf = subparsers.add_parser('pack_features')
    parser_pf.add_argument('--fe_dir', type=str)
    parser_pf.add_argument('--csv_path', type=str)
    parser_pf.add_argument('--out_path', type=str)
    
    parser_cs = subparsers.add_parser('calculate_scaler')
    parser_cs.add_argument('--hdf5_path', type=str)
    parser_cs.add_argument('--out_path', type=str)

    args = parser.parse_args()
    
    if args.mode == 'extract_features':
        extract_features(wav_dir=args.wav_dir, 
                         out_dir=args.out_dir, 
                         recompute=args.recompute)
    elif args.mode == 'pack_features':
        pack_features_to_hdf5_train(fe_dir=args.fe_dir, 
                              csv_path=args.csv_path, 
                              out_path=args.out_path)
    elif args.mode == 'calculate_scaler':
        calculate_scaler(hdf5_path=args.hdf5_path, 
                         out_path=args.out_path)
    else:
        raise Exception("Incorrect argument!")
