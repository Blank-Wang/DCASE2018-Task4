"""
SUMMARY:  DCASE 2017 task 4 io.
AUTHOR:   Qiuqiang Kong
Created:  2017.07.12
Modified: -
--------------------------------------
"""
import numpy as np
import csv
import os
import gzip
import pandas as pd
import vad


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_my_open(path):
    suffix = os.path.splitext(path)[-1]
    if suffix == '.gz':
        return gzip.open
    elif suffix == '.csv':
        return open
    else:
        raise Exception("Error!")


### Audio tagging (AT) related io. 
def at_write_prob_mat_to_csv(na_list, prob_mat, out_path):
    """Write out audio tagging (AT) prediction probability to a csv file. 
    
    Args:
      na_list: list of names. 
      prob_mat: ndarray, (n_clips, n_labels). 
      out_path: string, path to write out the csv file. 
      
    Returns:
      None
    """
    create_folder(os.path.dirname(out_path))
    f = gzip.open(out_path, 'w')
    for n in xrange(len(na_list)):
        na = na_list[n]
        f.write(na)
        for p in prob_mat[n]:
            f.write('\t' + "%.3f" % p)
        f.write('\r\n')
    f.close()

def at_read_prob_mat_csv(prob_mat_path):
    """Load prob_mat from csv.
    
    Args:
      prob_mat_path: string, path of AT prediction probability. each row in csv
          should be: [na][tab][prob1][tab][...][tab][probK]
           
    Returns:
      na_list: list of string. 
      prob_mat: ndarray, probabilites with shape (n_clips, n_labels)
    """
    my_open = get_my_open(prob_mat_path)
    with my_open(prob_mat_path, 'rb') as f_read:
        reader = csv.reader(f_read, delimiter='\t')
        lis = list(reader)
        
    na_list = []
    prob_mat = []
        
    for li in lis:
        na_list.append(li[0])
        p_ary = np.array([float(e) for e in li[1:]])
        prob_mat.append(p_ary)
        
    prob_mat = np.array(prob_mat)
    return na_list, prob_mat
    
def at_write_prob_mat_to_submission_csv(na_list, at_prob_mat, 
                                        lbs, thres_ary, out_path):
    """Write prob_mat to submission csv. 
    
    Args:
      na_list: list of strings. 
      at_prob_mat: ndarray with shape (n_clips, n_labels)
      lbs: list of strings. 
      thres_ary: list of float. 
      out_path: string, path to write out submission csv. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    
    f_write = open(out_path, 'w')
    for n in xrange(len(na_list)):
        na = na_list[n]
        prob_ary = at_prob_mat[n]
        indexes = []
        for j1 in xrange(len(lbs)):
            if prob_ary[j1] > thres_ary[j1]:
                indexes.append(j1)
        indexes = np.array(indexes)
        
        if len(indexes) == 0:
            indexes = [np.argmax(prob_ary)]
        for index in indexes:
            f_write.write(na + '\t' + str(0.000) + '\t' + str(10.000) + '\t' \
                          + lbs[index] + '\r\n')
    f_write.close()
    print("Write", out_path, "successfully!")
    
def at_read_submission_csv(submission_csv, lbs):
    """Read submission csv to prob_mat. 
    
    Args:
      submission_csv: string, path of submission csv. 
      lbs: list of string. 
      
    Returns:
      na_list: list of string. 
      digit_mat: ndarray, (n_clips, n_labels)
    """
    lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
    my_open = get_my_open(submission_csv)
    with my_open(submission_csv, 'rb') as f_read:
        reader = csv.reader(f_read, delimiter='\t')
        lis = list(reader)
        
    na_list = []
    digit_mat = []
    for li in lis:
        na = li[0]
        if len(li) > 1:
            [bgn_time, fin_time, lb] = li[1:]
            idx = lb_to_idx[lb]
            if na not in na_list:
                na_list.append(na)
                ary = np.zeros(len(lbs))
                ary[idx] = 1
                digit_mat.append(ary)
            else:
                digit_mat[na_list.index(na)][idx] = 1
        else:
            if na not in na_list:
                na_list.append(na)
                ary = np.zeros(len(lbs))
                digit_mat.append(ary)
        
    digit_mat = np.array(digit_mat)
    return na_list, digit_mat
    
def at_write_prob_mat_csv_to_submission_csv(at_prob_mat_path, lbs, thres_ary, out_path):
    """Write out submission csv from prob_mat csv. 
    
    Args:
      at_prob_mat_path: string, path of prob_mat csv file. 
      lbs: list, list of string. 
      thres_ary: list of float. 
      out_path: string, path of submission csv to write out. 
      
    Returns:
      None
    """
    create_folder(os.path.dirname(out_path))
    my_open = get_my_open(at_prob_mat_path)
    
    (na_list, prob_mat) = at_read_prob_mat_csv(at_prob_mat_path)
    at_write_prob_mat_to_submission_csv(na_list, prob_mat, lbs, thres_ary, out_path)
    
def at_read_weak_gt_csv(weak_gt_csv, lbs):
    """Read strong_gt csv to digit mat with shape (n_clips, n_labels). 
    
    Args:
      weak_gt_csv: string, strong_gt csv path. 
      lbs: list of string. 
      
    Returns:
      na_list: list of string. 
      digit_mat: ndarray, (n_clips, n_labels)
    """
    lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
#     with open(weak_gt_csv, 'rb') as f:
#         reader = csv.reader(f, delimiter='\t')
#         reader = csv.reader(f)
#         lis = list(reader)
    df = pd.read_csv(weak_gt_csv, sep='\t')
    lis=df.values
       
    na_list = []
    digit_mat = []
    for li in lis:        
        na = li[0]
        lb = li[1]
        lb = lb.split(',')
        for st in lb:
            
            if st not in lb_to_idx.keys():
                print(st, "not a key! (Please ignore)")
            else:
                index = lb_to_idx[st]
                if na not in na_list:
                    na_list.append(na)
                    gt_ary = np.zeros(len(lbs))
                    gt_ary[index] = 1
                    digit_mat.append(gt_ary)
                else:
                    digit_mat[na_list.index(na)][index] = 1
    digit_mat = np.array(digit_mat)
    return na_list, digit_mat
    
    
### Sound event detection (SED) related
def sed_write_prob_mat_list_to_csv(na_list, prob_mat_list, out_path):
    """Write out prob_mat_list to a csv file. 
    
    Args:
      na_list: list of string. 
      prob_mat_list: list of ndarray. Shape of each ndarray is (n_time, n_labels)
    
      out_path: string, path to write out the csv file. 
    Returns:
      None. 
    """
    f = gzip.open(out_path, 'w')
    for n in range(len(na_list)):
        na = na_list[n]
        prob_mat = prob_mat_list[n]
        (n_time, n_lb) = prob_mat.shape
        for i2 in xrange(n_time):
            f.write(na)
            for i3 in xrange(n_lb):
                f.write("\t%.4f" % prob_mat[i2, i3])
            f.write("\n")
    f.close()
    
def sed_read_prob_mat_list_to_csv(sed_prob_mat_list_path):
    """Read prob_mat_list csv. 
    
    Args:
      sed_prob_mat_list_path: string, path of prob_mat_list. 
           
    Returns:
      na_list: list of string. 
      prob_mat_list: list of ndarray. Shape of each ndarray is (n_time, n_labels)
    """
    my_open = get_my_open(sed_prob_mat_list_path)
    with my_open(sed_prob_mat_list_path, 'rb') as f_read:
        reader = csv.reader(f_read, delimiter='\t')
        lis = list(reader)
    
    na_list = []
    prob_mat_list = []
    prev_na = ''
    
    prob_mat = []
    for li in lis:
        na = li[0]
        prob_ary = np.array([float(e) for e in li[1:]])
        
        if na != prev_na:
            if prev_na == '':
                pass
            else:
                prob_mat = np.array(prob_mat)
                prob_mat_list.append(prob_mat)
                prob_mat = []
                
            na_list.append(na)
            prob_mat.append(prob_ary)
        else:
            prob_mat.append(prob_ary)
        prev_na = na
    prob_mat = np.array(prob_mat)
    prob_mat_list.append(prob_mat)
    
    return na_list, prob_mat_list
    
def sed_write_prob_mat_list_to_submission_csv(na_list, prob_mat_list, lbs, 
                                              thres_ary, step_sec, out_path):
    """Write prob_mat_list to submission csv. 
    
    Args:
      na_list: list of string. 
      prob_mat_list: list of ndarray. Shape of each ndarray is (n_time, n_labels)
      lbs: list of string. 
      thres_ary: list of float. 
      step_sec: float, step duration in second. 
      out_path: string, path to write out submission csv. 
      
    Returns:
      None. 
    """
    create_folder(os.path.dirname(out_path))
    f = open(out_path, 'w')
    cnt = 0
    for n in xrange(len(na_list)):
        na = na_list[n]
        prob_mat = prob_mat_list[n]
        flag = False
        for i2 in xrange(len(lbs)):
            event_list = vad.activity_detection(x=prob_mat[:, i2], 
                                                thres=thres_ary[i2], 
                                                n_smooth=10, 
                                                n_salt=10)
            if len(event_list) != 0:
                flag = True
                for [bgn, fin] in event_list:
                    bgn_sec = step_sec * bgn
                    fin_sec = step_sec * fin
                    f.write(na + "\t" + str(bgn_sec) + "\t" + \
                            str(fin_sec) + "\t" + lbs[i2] + "\n")
        if flag == False: 
            f.write(na + "\n")
    f.close()
    print("Write", out_path, "successfully!")
    
def sed_read_submission_csv_to_prob_mat_list(submission_csv, lbs, step_sec, max_len):
    """Read submission csv to prob_mat_list. 
    
    Args: 
      submission_csv: string, submission csv path. 
      step_sec: float, step duration in second. 
      max_len: int, maximum length of sequence. 
      
    Returns:
      na_list: list of string. 
      digit_mat_list: list of ndarray. Shape of each ndarray is (n_time, n_labels)
    """
    lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
    my_open = get_my_open(submission_csv)
    with my_open(submission_csv, 'rb') as f_read:
        reader = csv.reader(f_read, delimiter='\t')
        lis = list(reader)
        
    na_list = []
    digit_mat_list = []
    for li in lis:
        na = li[0]
        if len(li) > 1:
            bgn_time = float(li[1])
            fin_time = float(li[2])
            lb = li[3]
            idx = lb_to_idx[lb]
            bgn_frame = int(np.floor(bgn_time / step_sec))
            fin_frame = int(min(np.ceil(fin_time / step_sec), max_len))
            if na not in na_list:
                na_list.append(na)
                prob_mat = np.zeros((max_len, len(lbs)))
                prob_mat[bgn_frame: fin_frame, idx] = 1.
                digit_mat_list.append(prob_mat)
            else:
                digit_mat_list[na_list.index(na)][bgn_frame: fin_frame, idx] = 1
        else:
            na_list.append(na)
            prob_mat = np.zeros((max_len, len(lbs)))
            digit_mat_list.append(prob_mat)
            
    return na_list, digit_mat_list
    
def sed_write_prob_mat_list_csv_to_submission_csv(sed_prob_mat_list_path, lbs, thres_ary, 
                                        step_sec, out_path):
    """Write out prob_mat_list to submission csv. 
    
    Args:
      sed_prob_mat_list_path: string, path of prob_mat_list. 
      lbs: list of string. 
      thres_ary: list of float. 
      step_sec: float, step duration in second. 
      out_path: string, path to write out submission csv. 
      
    Returns:
      None. 
    """
    (na_list, prob_mat_list) = sed_read_prob_mat_list_to_csv(sed_prob_mat_list_path)
    sed_write_prob_mat_list_to_submission_csv(na_list, prob_mat_list, lbs, 
                                              thres_ary, step_sec, out_path)
    
def sed_read_strong_gt_csv(strong_gt_csv, lbs, step_sec, max_len):
    """Read strong_gt csv to prob_mat_list. 
    
    Args: 
      strong_gt_csv: string, path of strong_gt csv. 
      lbs: list of string. 
      step_sec: float, step duration in second. 
      max_len: int, maximum length of sequence. 
      
    Returns: 
      na_list: list of string. 
      digit_mat_list: list of ndarray. Shape of each ndarray is (n_time, n_labels)
    """
    lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
    
#     with open(strong_gt_csv, 'rb') as f:
# #         reader = csv.reader(f, delimiter='\t')
#         reader = csv.reader(f)
#         lis = list(reader)

    df = pd.read_csv(strong_gt_csv, sep='\t')
    lis=df.values
        
    na_list = []
    digit_mat_list = []
    cnt = 0
    for li in lis:
        cnt += 1
        na = li[0]
        bgn = float(li[1])
        fin = float(li[2])
        lb = li[3]
        
        if lb not in lb_to_idx.keys():
            print( lb, "not a key! (Please ignore)")
        else:
            idx = lb_to_idx[lb]
            bgn_frame = int(np.floor(bgn / step_sec))
            fin_frame = min(int(fin / step_sec), max_len)
            if na not in na_list:
                na_list.append(na)
                prob_mat = np.zeros((max_len, len(lbs)))
                prob_mat[bgn_frame: fin_frame, idx] = 1.
                digit_mat_list.append(prob_mat)
            else:
                digit_mat_list[na_list.index(na)][bgn_frame: fin_frame, idx] = 1
        
    return na_list, digit_mat_list