#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Summary:  DCASE 2018 task 4 Large-scale weakly supervised sound event detection.
Author: Dezhi Wang, wang_dezhi@hotmail.com, NUDT - (thanks to Qiuqiang Kong, q.kong@surrey.ac.uk, CVSSP, University of Surrey)
Modified: 31/10/2018
"""
from __future__ import print_function 
import sys
import cPickle
import numpy as np
import argparse
import glob
import time
import os
from scipy.interpolate import interp1d
from distutils.dir_util import copy_tree
import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random

import config as cfg
from prepare_data_1017 import create_folder, load_hdf5_data, do_scale_3CH, calculate_scaler_3CH, pack_features_to_hdf5_test, pack_features_to_hdf5_train, extract_features_parallel
from data_generator import RatioDataGenerator
from data_generator_mixup import RatioDataGeneratorMixup
from evaluation import evaluate
from evaluation import io_config
import shutil
from resnet50_Mod_Nopre import *
from resnet50_Mod2 import *
from xception_Mod import *
from Sed_Evaluation_EVA import sed_event_eval2
from keras.utils import multi_gpu_model

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

def filetime(file):
    stat_file = os.stat(file)
    last_mod_time = stat_file.st_mtime
    return last_mod_time

# Train model
def train(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args['tr_hdf5_path'], verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args['te_hdf5_path'], verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))

    # Scale data
    tr_x = do_scale_3CH(tr_x, args['scaler_path'], verbose=1)
    te_x = do_scale_3CH(te_x, args['scaler_path'], verbose=1)
    
    # Build model
    (_, n_time, n_freq, naxis) = tr_x.shape    
    input_logmel = Input(shape=(n_time, n_freq, naxis), name='in_layer')    
    
    if args['ResNet50']:
        model_resnet=ResNet50_Mod2(include_top=False, weights='imagenet', input_tensor=input_logmel, input_shape=None,
                              pooling='max',  classes=num_classes)
        a1=model_resnet.get_layer('avg_pool').output
        a1=Dropout(0.5)(a1) 
        a1 = Reshape((160, 2048))(a1)
        a1 = Convolution1D(128,3,padding='same')(a1)
        
    if args['Xception']:
        model_xception=Xception_Mod(include_top=False, weights='imagenet', input_tensor=input_logmel, input_shape=None,
                              pooling='max',  classes=num_classes)
        a1=model_xception.get_layer('block14_sepconv2_act').output 
        a1 = MaxPooling2D(pool_size=(1, 2))(a1) 
        a1=Dropout(0.5)(a1) 
        a1 = Reshape((160, 2048))(a1)
        a1 = Convolution1D(128,3,padding='same')(a1)

#     model_resnet=ResNet50_Mod_Nopre(include_top=False, weights=None, input_tensor=input_logmel, input_shape=None,
#                           pooling='avg',  classes=num_classes)
#     a1=model_resnet.get_layer('avg_pool').output  
#     a1 = Reshape((480, 256))(a1)
       
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='tanh', return_sequences=True, dropout=0.3))(a1) 
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True, dropout=0.3))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    a2=Dropout(0.3)(a2) 
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    opt=keras.optimizers.adam(lr=args['learning_rate']) #default lr=0.001
    
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    # Save model callback
    filepath = os.path.join(args['out_model_dir'], "Model-<{epoch:02d}><{loss:2.4f}><{acc:.4f}><{val_loss:2.4f}><{val_acc:.4f}>.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
#     gen = RatioDataGenerator(batch_size=args['batchsize_train'], type='train') #without mixup
    gen = RatioDataGeneratorMixup(batch_size=args['batchsize_train'], type='train', alpha=args['Mixup_alpha'], mixup=True) #with mixup

    # load trained model
    if args['load_pretrain_weights']:
        model.load_weights(args['pretrain_weights'])
    
    # Train
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=args['batchfile_num'],      #int(tr_y.shape[0] / args['batchsize_train'])
                        epochs=args['epoch_num'],              # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))

# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in xrange(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all

# Process the test dataset
# Recognize and write probabilites. 
def recognize(args, at_bool, sed_bool):
    (te_x, te_y, te_na_list) = load_hdf5_data(args['te_hdf5_path'], verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale_3CH(x, args['scaler_path'], verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    
    iterms = os.listdir(args['model_dir'])
    iterms = sorted(iterms,key= lambda x:filetime(os.path.join(args['model_dir'],x)),reverse=True)
    print(iterms)
    
    
    for epoch in range(args['epoch_range']):
        t1 = time.time()
        model_path = glob.glob(os.path.join(args['model_dir'], iterms[epoch]))
        model = load_model(model_path[0])
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x)
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func = K.function([in_layer.input, K.learning_phase()], 
                              [loc_layer.output])
            pred3d = run_func(func, x, batch_size=args['batchsize_eva'])
            fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_config.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args['out_dir'], "at_prob_mat.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        if args['timealign']:
            fusion_sed_list=timealign(fusion_sed_list, args['timesteps'])
        if args['max_ensemble']:
            fusion_sed = np.max(np.array(fusion_sed_list), axis=0)
        else:
            fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        io_config.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args['out_dir'], "sed_prob_mat_list.csv.gz"))
            
    print("Prediction for Test Dataset finished!")

def timealign(fusion_sed_list, timesteps):
    for i in range(len(fusion_sed_list)):
        files, steps, classes = fusion_sed_list[i].shape
        if steps != timesteps:
            arraytmp=np.zeros((files, timesteps, classes))
            for j in range(files):
                for k in range(classes):
                    xx = np.linspace(0, 10, num=steps, endpoint=True)
                    xx2 = np.linspace(0, 10, num=timesteps, endpoint=True)
                    yy = fusion_sed_list[i][j,:,k]
                    f1 = interp1d(xx, yy, kind='nearest')
                    yy2 = f1(xx2)
                    arraytmp[j,:,k]=yy2
            fusion_sed_list[i]=arraytmp
    return fusion_sed_list          
    
# Get stats from probabilites. 
def get_stat(args, at_bool, sed_bool):
    lbs = cfg.lbs
    step_time_in_sec = 10.0/args['timesteps']
    max_len = args['timesteps']
    thres_ary = args['threshold']
    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args['pred_dir'], "at_prob_mat.csv.gz")
        at_stat_path = os.path.join(args['stat_dir'], "at_stat.csv")
        at_submission_path = os.path.join(args['submission_dir'], "at_submission-{0}.csv".format(args['defined_name']))
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv="./metadata/test/test_file_available_weak.csv", 
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_config.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args['pred_dir'], "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args['stat_dir'], "sed_stat.csv")
        sed_submission_path = os.path.join(args['submission_dir'], "sed_submission-{0}.csv".format(args['defined_name']))
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv="./metadata/test/test_file_available_strong.csv", 
            lbs=lbs, 
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_config.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat for Test Dataset finished!")

# Process the eval dataset
def recognize_eval(args, at_bool, sed_bool):
    (eval_x, eval_y, eval_na_list) = load_hdf5_data(args['eval_hdf5_path'], verbose=1)
    x = eval_x
    y = eval_y
    na_list = eval_na_list
    
    x = do_scale_3CH(x, args['scaler_path'], verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    
    iterms = os.listdir(args['model_dir'])
    iterms = sorted(iterms,key= lambda x:filetime(os.path.join(args['model_dir'],x)),reverse=True)
    print(iterms)
    
    
    for epoch in range(args['epoch_range']):
        t1 = time.time()
        model_path = glob.glob(os.path.join(args['model_dir'], iterms[epoch]))
        model = load_model(model_path[0])
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x)
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func = K.function([in_layer.input, K.learning_phase()], 
                              [loc_layer.output])
            pred3d = run_func(func, x, batch_size=args['batchsize_eva'])
            fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_config.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args['out_dir'], "at_prob_mat_eval.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        if args['timealign']:
            fusion_sed_list=timealign(fusion_sed_list, args['timesteps'])
        if args['max_ensemble']:
            fusion_sed = np.max(np.array(fusion_sed_list), axis=0)
        else:
            fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        io_config.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args['out_dir'], "sed_prob_mat_list_eval.csv.gz"))
            
    print("Prediction for Evaluation Dataset finished!")

# Get stats from probabilites. 
def get_stat_eval(args, at_bool, sed_bool):
    lbs = cfg.lbs
    step_time_in_sec = 10.0/args['timesteps']
    max_len = args['timesteps']
    thres_ary = args['threshold']

    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args['pred_dir'], "at_prob_mat_eval.csv.gz")
        at_stat_path = os.path.join(args['stat_dir'], "at_stat_eval.csv")
        at_submission_path = os.path.join(args['submission_dir'], "at_submission_eval-{0}.csv".format(args['defined_name']))
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv="./metadata/eval/eval_file_available_weak_labels.csv", 
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_config.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args['pred_dir'], "sed_prob_mat_list_eval.csv.gz")
        sed_stat_path = os.path.join(args['stat_dir'], "sed_stat_eval.csv")
        sed_submission_path = os.path.join(args['submission_dir'], "sed_submission_eval-{0}.csv".format(args['defined_name']))
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv="./metadata/eval/eval_file_available_strong_labels.csv", 
            lbs=lbs, 
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_config.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat for Evaluation Dataset finished!")

if __name__ == '__main__':

    worklocation='.'
    alarm=0.25;blender=0.25;cat=0.25;dishes=0.25;dog=0.25;electric=0.25;frying=0.25;water=0.25;speech=0.25;vacuum=0.25; 
    args={'mode':'train', 'tr_hdf5_path':worklocation+'/packed_features_3CH/logmel/training_including_indomain.h5',
          'eval_hdf5_path':worklocation+'/packed_features_3CH/logmel/evaluation.h5', 'te_hdf5_path':worklocation+'/packed_features_3CH/logmel/testing.h5',
          'scaler_path':worklocation+'/scalers_3CH/logmel/training.scaler', 'out_model_dir': worklocation+'/models/crnn_sed_3CH','model_dir':worklocation+'/models/crnn_sed_3CH',
          'out_dir':worklocation+'/preds/crnn_sed_3CH','pred_dir':worklocation+'/preds/crnn_sed_3CH','stat_dir':worklocation+'/stats_3CH/crnn_sed',
          'submission_dir':worklocation+'/submissions/crnn_sed', 'epoch_num':10, 'batchsize_train':18, 'batchfile_num':500,'batchsize_eva':20,
          'learning_rate':0.001, 'epoch_range':3, 'timealign': False, 'timesteps': 160, 'max_ensemble': False, 'ResNet50':True, 'Xception':False,
          'load_models': True, 'generate_submissions': True, 'Mixup_alpha':0.2, 'defined_name':'0.95-0-0',
          'load_pretrain_weights':False, 
          'pretrain_weights':
          './models/ResNet-weak/Resnet-Pretrain__0.0784__0.8758__0.2725__0.9385_.hdf5',
          'threshold':[speech, dog, cat, alarm, dishes, frying, blender, water, vacuum, electric]
           }
    
##########################Extract & Pack Features##############################    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # You need to modify to your dataset path
    TEST_WAV_DIR="/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/test"
    TRAIN_WAV_DIR_Weak_Indomain="/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/train/weak_and_indomain"
    TRAIN_WAV_DIR="/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/train/weak"
    TRAIN_WAV_DIR_InDomain="/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/train/unlabel_in_domain"
    TRAIN_WAV_DIR_OutOfDomain="/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/train/unlabel_out_of_domain"
    EVALUATION_WAV_DIR="/media/user/Duty/Data_Bank/DCASE18-TASK4-Dataset/audio/eval"
     
    # You can to modify to your own workspace.  
    create_folder(args['out_model_dir'])
    create_folder("./features_3CH/logmel/testing")
    create_folder("./features_3CH/logmel/training_weak_indomain")
    create_folder("./features_3CH/logmel/evaluation")
    create_folder("./packed_features_3CH/logmel")
    create_folder("./scalers_3CH/logmel") 
    create_folder('./stats_3CH/crnn_sed')
    
    # Extract features parallel
    extract_features_parallel(TEST_WAV_DIR,"./features_3CH/logmel/testing",recompute=False)
    extract_features_parallel(EVALUATION_WAV_DIR,"./features_3CH/logmel/evaluation",recompute=False)
    extract_features_parallel(TRAIN_WAV_DIR_Weak_Indomain,"./features_3CH/logmel/training_weak_indomain",recompute=False)  

    # Pack features
    pack_features_to_hdf5_train("./features_3CH/logmel/training_weak_indomain", "metadata/train/train_file_including_indomain-{0}.csv".format(args['defined_name']), 
                                                "./packed_features_3CH/logmel/training_including_indomain.h5") 
    pack_features_to_hdf5_test("./features_3CH/logmel/testing", "metadata/test/test_file_available_weak.csv", "./packed_features_3CH/logmel/testing.h5") 
    pack_features_to_hdf5_test("./features_3CH/logmel/evaluation", "" , "./packed_features_3CH/logmel/evaluation.h5") 
     
    # calsulate the scaler
    calculate_scaler_3CH("./packed_features_3CH/logmel/training_including_indomain.h5", args['scaler_path'])
   
    args['mode'] = 'train'
    if os.path.exists(args['out_model_dir']):
        shutil.rmtree(args['out_model_dir'])
    os.makedirs(args['out_model_dir'])
    if os.path.exists(args['out_dir']):
        shutil.rmtree(args['out_dir'])
    os.makedirs(args['out_dir']) 
    if os.path.exists(args['stat_dir']):
        shutil.rmtree(args['stat_dir'])
    os.makedirs(args['stat_dir'])  
    train(args)
      
    # process the test dataset
    args['mode'] = 'recognize'
    if args['load_models']:
        recognize(args, at_bool=True, sed_bool=True)
    args['mode'] = 'get_stat' 
    get_stat(args, at_bool=True, sed_bool=True)
       
    # process the evaluation dataset
    args['mode'] = 'recognize_eval'
    if args['generate_submissions']:
        recognize_eval(args, at_bool=True, sed_bool=True)
        args['mode'] = 'get_stat_eval'
        get_stat_eval(args, at_bool=True, sed_bool=True)
        
#     evaluate the SED result using sed_eval package
    sed_event_eval2(ref_file='./metadata/test/test_file_available_strong.csv', est_file='./submissions/crnn_sed/sed_submission-{0}.csv'.format(args['defined_name']))
    
    sed_event_eval2(ref_file='./metadata/eval/eval_ground_truth.csv', est_file='./submissions/crnn_sed/sed_submission_eval-{0}.csv'.format(args['defined_name']))
    
    print('Evaluate the Eval_file:'+'sed_submission_eval-{0}.csv'.format(args['defined_name']))
    print('Computation Finished~')
   
   
   
