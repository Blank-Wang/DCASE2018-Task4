"""
SUMMARY:  Evaluation of DCASE 2017 Task 4. Including audio tagging (AT) and
          Sound event detection (SED) criteria. 
AUTHOR:   Qiuqiang Kong
Created:  2017.06.04
Modified: 2017.07.12
--------------------------------------
"""
import os
import sys
import csv
import gzip
import numpy as np
# import cPickle
import pickle

import evaluation as eva
from io_config import create_folder, get_my_open
from io_config import at_read_prob_mat_csv, at_read_submission_csv, at_read_weak_gt_csv
from io_config import sed_read_prob_mat_list_to_csv, sed_read_submission_csv_to_prob_mat_list, sed_read_strong_gt_csv


class AudioTaggingEvaluate(object):
    """
    Methods:
      get_stats_from_prob_mat_csv()
      get_stats_from_prob_mat()
      get_stats_from_submit_format()
      write_out_ankit_stat()
      write_stat_to_csv()
      print_stat()
    """
    def __init__(self, weak_gt_csv, lbs):
        self.weak_gt_csv = weak_gt_csv
        self.lbs = lbs
        self.lb_to_idx = {lb: index for index, lb in enumerate(self.lbs)}
        
    def get_stats_from_prob_mat_csv(self, pd_prob_mat_csv, thres_ary):
        """Get stats from prob_mat csv and ground truth csv. 
        
        Args:
          pd_prob_mat_csv: string, path of prob_mat csv. 
          thres_ary: list of float | 'auto' | 'no_need'. 
        
        Returns:
          stat. 
        """
        (gt_na_list, gt_mat) = at_read_weak_gt_csv(self.weak_gt_csv, self.lbs)
        (pd_na_list, pd_prob_mat) = at_read_prob_mat_csv(pd_prob_mat_csv)
        pd_prob_mat = self._reorder_mat(pd_prob_mat, pd_na_list, gt_na_list)
        del pd_na_list
        
        stat = self.get_stats_from_prob_mat(pd_prob_mat, gt_mat, thres_ary)
        return stat
        
    def get_stats_from_prob_mat(self, pd_prob_mat, gt_mat, thres_ary):
        """Get stats from prob_mat and ground truth mat. 
        
        Args:
          pd_prob_mat: ndarray, (n_clips, n_labels)
          gt_prob_mat: ndarray, (n_clips, n_labels)
          thres_ary: list of float | 'auto' | 'no_need'. 
          
        Returns:
          stat. 
        """        
        n_lbs = len(self.lbs)

        stat = {}
        if type(thres_ary) is list:
            stat['thres_ary'] = thres_ary
        elif thres_ary == 'auto':
            thres_ary = self._get_best_thres_ary(pd_prob_mat, gt_mat)
            stat['thres_ary'] = thres_ary
        elif thres_ary == 'no_need':
            thres_ary = [0.5] * len(self.lbs)
            stat['thres_ary'] = ['no_need'] * len(self.lbs)
        else:
            raise Exception("thres_ary type wrong!")
        
        pd_digit_mat = self._get_digit_mat_from_thres_ary(pd_prob_mat, thres_ary)
        
        # overall stat
        eer = eva.eer(pd_prob_mat.flatten(), gt_mat.flatten())
        auc = eva.roc_auc(pd_prob_mat.flatten(), gt_mat.flatten())
        (tp, fn, fp, tn) = eva.tp_fn_fp_tn(pd_digit_mat, gt_mat, 0.5)
        prec = eva.precision(pd_digit_mat, gt_mat, 0.5)
        rec = eva.recall(pd_digit_mat, gt_mat, 0.5)
        f_val = eva.f_value(prec, rec)
        stat['overall'] = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 
                             'precision': prec, 'recall': rec, 'f_value': f_val, 
                             'eer': eer, 'auc': auc}

        # element-wise stat
        stat['event_wise'] = {}
        for k in xrange(len(self.lbs)):
            eer = eva.eer(pd_prob_mat[:, k], gt_mat[:, k])
            auc = eva.roc_auc(pd_prob_mat[:, k], gt_mat[:, k])
            (tp, fn, fp, tn) = eva.tp_fn_fp_tn(pd_digit_mat[:, k], gt_mat[:, k], 0.5)
            prec = eva.precision(pd_digit_mat[:, k], gt_mat[:, k], 0.5)
            rec = eva.recall(pd_digit_mat[:, k], gt_mat[:, k], 0.5)
            f_val = eva.f_value(prec, rec)
            stat['event_wise'][self.lbs[k]] = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 
                             'precision': prec, 'recall': rec, 'f_value': f_val, 
                             'eer': eer, 'auc': auc}
        return stat
        
    def get_stats_from_submit_format(self, submission_csv):
        """Get stats from submission csv and ground truth csv. 
        """
        (gt_na_list, gt_mat) = at_read_weak_gt_csv(self.weak_gt_csv, self.lbs)
        (pd_na_list, pd_digit_mat) = at_read_submission_csv(submission_csv, self.lbs)
        pd_digit_mat = self._reorder_mat(pd_digit_mat, pd_na_list, gt_na_list)
        del pd_na_list
        
        stat = self.get_stats_from_prob_mat(pd_digit_mat, gt_mat, thres_ary='no_need')
        return stat
    
    def write_out_ankit_stat(self, submission_csv, ankit_csv, stat_path):
        """Write out stats using ankit's evaluation. 
        """
        create_folder(os.path.dirname(stat_path))
        
        f = open(stat_path, 'w')
        f.write("# --------- Ankitshah009's Evaluation ---------\n")
        f.close()
        
        gtDS = FileFormat(ankit_csv)
        pdDS = FileFormat(submission_csv)
        gtDS.computeMetrics(pdDS, stat_path)    # write to stat csv
        
        with open(stat_path, 'rb') as f:
            reader = csv.reader(f, delimiter='\t')
            lis = list(reader)
    
    def write_stat_to_csv(self, stat, stat_path, mode='w'):
        """Write out stats. 
        """
        create_folder(os.path.dirname(stat_path))
        f = open(stat_path, mode)
        f.write('\n')
        f.write("\n\n# --------- Event wise evaluation ---------\n")
        f.write("\t")
        for lb in self.lbs:
            f.write("\t" + lb[0:6])
            
        f.write("{0:12}".format("\nthres:"))
        for i1 in xrange(len(self.lbs)):
            f.write("\t" + "%s" % stat['thres_ary'][i1])
            
        f.write("{0:12}".format("\ntp:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['tp'])
        
        f.write("{0:12}".format("\nfn:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['fn'])

        f.write("{0:12}".format("\nfp:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['fp'])
            
        f.write("{0:12}".format("\ntn:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['tn'])
            
        f.write("{0:12}".format("\nprecision:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['precision'])
            
        f.write("{0:12}".format("\nrecall:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['recall'])
        
        f.write("{0:12}".format("\nf_value:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['f_value'])
            
        f.write("{0:12}".format("\neer:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['eer'])
            
        f.write("{0:12}".format("\nauc:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['auc'])
                    
        f.write("\n")
        f.write("{0:12}".format("\nmean precision:") + "\t%.3f" % self._get_mean_stat(stat, 'precision'))
        f.write("{0:12}".format("\nmean recall:") + "\t%.3f" % self._get_mean_stat(stat, 'recall'))
        f.write("{0:12}".format("\nmean f_value:") + "\t%.3f" % self._get_mean_stat(stat, 'f_value'))
        f.write("{0:12}".format("\nmean eer:") + "\t%.3f" % self._get_mean_stat(stat, 'eer'))
        f.write("{0:12}".format("\nmean auc:") + "\t%.3f" % self._get_mean_stat(stat, 'auc'))
    
        f.write("\n\n# --------- Total evaluation (Mix all events together) ---------")
        f.write("{0:16}".format("\ntotal tp:") + "\t%d" % stat['overall']['tp'])
        f.write("{0:16}".format("\ntotal fn:") + "\t%d" % stat['overall']['fn'])
        f.write("{0:16}".format("\ntotal fp:") + "\t%d" % stat['overall']['fp'])
        f.write("{0:16}".format("\ntotal tn:") + "\t%d" % stat['overall']['tn'])
        f.write("\n")
        f.write("{0:18}".format("\ntotal precision:") + "\t%.3f" % stat['overall']['precision'])
        f.write("{0:18}".format("\ntotal recall:") + "\t%.3f" % stat['overall']['recall'])
        f.write("{0:18}".format("\ntotal f_value:") + "\t%.3f" % stat['overall']['f_value'])
        f.write("\n")
        f.close()
        print("Write out stat to", stat_path, "successfully!")
        
    def print_stat(self, stat_path):
        """Print out stat csv file. 
        """
        fr = open(stat_path, 'r')
        print(fr.read())
        fr.close()

    def _reorder_mat(self, pd_mat, pd_na_list, gt_na_list):
        indexes = [pd_na_list.index(e) for e in gt_na_list]
        return pd_mat[indexes]
        
    def _get_best_thres_ary(self, pd_prob_mat, gt_mat):
        thres_ary = []
        for k in xrange(len(self.lbs)):
            f_val_max = -np.inf
            best_thres = None
            for thres in np.arange(0., 1.+1e-6, 0.01):
                prec = eva.precision(pd_prob_mat[:, k], gt_mat[:, k], thres)
                rec = eva.recall(pd_prob_mat[:, k], gt_mat[:, k], thres)
                f_val = eva.f_value(prec, rec)
                if f_val > f_val_max:
                    f_val_max = f_val
                    best_thres = thres
            thres_ary.append(best_thres)
        return thres_ary
        
    def _get_digit_mat_from_thres_ary(self, pd_prob_mat, thres_ary):
        pd_digit_mat = np.zeros_like(pd_prob_mat)
        for n in xrange(len(pd_prob_mat)):
            for k in xrange(len(self.lbs)):
                if pd_prob_mat[n, k] > thres_ary[k]:
                    pd_digit_mat[n, k] = 1
            if np.sum(pd_digit_mat[n, :]) == 0:
                pd_digit_mat[n, np.argmax(pd_prob_mat[n, :])] = 1
        return pd_digit_mat

    def _get_mean_stat(self, stat, type):
        val_ary = []
        for lb in self.lbs:
            val_ary.append(stat['event_wise'][lb][type])
        return np.mean(val_ary)
        
    
class SoundEventDetectionEvaluate(object):
    """
    Methods:
      get_stats_from_prob_mat_list_csv()
      get_stats_from_prob_mat_list()
      get_stats_from_submit_format()
      write_stat_to_csv()
      print_stat()
    """
    def __init__(self, strong_gt_csv, lbs, step_sec, max_len):
        self.strong_gt_csv = strong_gt_csv
        self.lbs = lbs
        self.lb_to_idx = {lb: index for index, lb in enumerate(self.lbs)}
        self.step_sec = step_sec
        self.max_len = max_len
        
    def get_stats_from_prob_mat_list_csv(self, pd_prob_mat_list_csv, thres_ary):
        """Get stats from prob_mat_list csv. 
        
        Args:
          pd_prob_mat_list_csv: string, path of prob_mat_list csv. 
          thres_ary: list of float | 'no_need'
        
        Returns:
          stat. 
        """
        (gt_na_list, gt_mat_list) = sed_read_strong_gt_csv(
            self.strong_gt_csv, self.lbs, self.step_sec, self.max_len)
        (pd_na_list, pd_prob_mat_list) = sed_read_prob_mat_list_to_csv(
            pd_prob_mat_list_csv)
        pd_prob_mat_list = self._reorder_mat_list(
            pd_prob_mat_list, pd_na_list, gt_na_list)
        del pd_na_list
        
        stat = self.get_stats_from_prob_mat_list(
                     pd_prob_mat_list, gt_mat_list, thres_ary)
        return stat
        
    def get_stats_from_prob_mat_list(self, pd_prob_mat_list, gt_mat_list, thres_ary):
        """Get stats from prob_mat_list and gt_mat_list. 
        
        Args:
          pd_prob_mat_list: list of ndarray. 
          gt_mat_list: list of ndarray. 
          thres_ary: list of float | 'no_need'. 
          
        Returns:
          stat. 
        """
        stat = {}
        if type(thres_ary) is list:
            stat['thres_ary'] = thres_ary
        elif thres_ary == 'no_need':
            thres_ary = [0.5] * len(self.lbs)
            stat['thres_ary'] = ['no_need'] * len(self.lbs)
        else:
            raise Exception("Incorrect thres_ary!")
        
        N = len(pd_prob_mat_list)        
        pd_digit_mat_list = []
        for n in xrange(N):
            pd_prob_mat = pd_prob_mat_list[n]
            pd_digit_mat = self._get_digit_mat_from_thres_ary(pd_prob_mat, thres_ary)
            pd_digit_mat_list.append(pd_digit_mat)
                   
        dict_list = []
        for n in xrange(N):
            pd_digit_mat = pd_digit_mat_list[n]
            gt_mat = gt_mat_list[n]
            dict = self._get_stats_from_digit_mat(pd_digit_mat, gt_mat)
            dict_list.append(dict)
            
        # Overall stat
        stat['overall'] = {}
        stat['overall']['tp'] = 0
        stat['overall']['fn'] = 0
        stat['overall']['fp'] = 0
        stat['overall']['tn'] = 0

        error_rate_ary = []
        for n in xrange(N):
            stat['overall']['tp'] += dict_list[n]['overall']['tp']
            stat['overall']['fn'] += dict_list[n]['overall']['fn']
            stat['overall']['fp'] += dict_list[n]['overall']['fp']
            stat['overall']['tn'] += dict_list[n]['overall']['tn']
            error_rate_ary.append(dict_list[n]['overall']['error_rate'])
            
        stat['overall']['precision'] = self._precision(stat['overall']['tp'], 
                                                         stat['overall']['fn'], 
                                                         stat['overall']['fp'], 
                                                         stat['overall']['tn'])
        stat['overall']['recall'] = self._recall(stat['overall']['tp'], 
                                                   stat['overall']['fn'], 
                                                   stat['overall']['fp'], 
                                                   stat['overall']['tn'])
        stat['overall']['f_value'] = eva.f_value(stat['overall']['precision'], 
                                                   stat['overall']['recall'])
        stat['overall']['error_rate'] = np.mean(np.array(error_rate_ary))

        # Event wise stat
        stat['event_wise'] = {}
        for lb in self.lbs:
            stat['event_wise'][lb] = {}
            stat['event_wise'][lb]['tp'] = 0
            stat['event_wise'][lb]['fn'] = 0
            stat['event_wise'][lb]['fp'] = 0
            stat['event_wise'][lb]['tn'] = 0
            
        for lb in self.lbs:
            for n in xrange(N):
                stat['event_wise'][lb]['tp'] += dict_list[n]['event_wise'][lb]['tp']
                stat['event_wise'][lb]['fn'] += dict_list[n]['event_wise'][lb]['fn']
                stat['event_wise'][lb]['fp'] += dict_list[n]['event_wise'][lb]['fp']
                stat['event_wise'][lb]['tn'] += dict_list[n]['event_wise'][lb]['tn']
                
            stat['event_wise'][lb]['precision'] = self._precision(
                                                        stat['event_wise'][lb]['tp'], 
                                                        stat['event_wise'][lb]['fn'], 
                                                        stat['event_wise'][lb]['fp'], 
                                                        stat['event_wise'][lb]['tn'])
            stat['event_wise'][lb]['recall'] = self._recall(
                                                        stat['event_wise'][lb]['tp'], 
                                                        stat['event_wise'][lb]['fn'], 
                                                        stat['event_wise'][lb]['fp'], 
                                                        stat['event_wise'][lb]['tn'])
            stat['event_wise'][lb]['f_value'] = eva.f_value(
                                                      stat['event_wise'][lb]['precision'], 
                                                      stat['event_wise'][lb]['recall'])
        return stat

    def get_stats_from_submit_format(self, submission_csv):
        """Get stats from submission csv. 
        
        Args:
          submission_csv: string, path of submission csv. 
          
        Returns:
          stat. 
        """
        (gt_na_list, gt_mat_list) = sed_read_strong_gt_csv(
            self.strong_gt_csv, self.lbs, self.step_sec, self.max_len)
        (pd_na_list, pd_digit_mat_list) = sed_read_submission_csv_to_prob_mat_list(
            submission_csv, self.lbs, self.step_sec, self.max_len)
        pd_digit_mat_list = self._reorder_mat_list(
            pd_digit_mat_list, pd_na_list, gt_na_list)
        del pd_na_list

        stat = self.get_stats_from_prob_mat_list(
                     pd_digit_mat_list, gt_mat_list, thres_ary='no_need')
        return stat

    def write_stat_to_csv(self, stat, stat_path):
        """Write out stat to stat csv. 
        
        Args:
          stat_path: string, path of stat csv to write out. 
        """
        create_folder(os.path.dirname(stat_path))
        f = open(stat_path, 'w')
        f.write('\n')
        f.write("\n\n# --------- Event wise evaluation ---------\n")
        f.write("\t")
        for lb in self.lbs:
            f.write("\t" + lb[0:6])
        
        f.write("{0:12}".format("\nthres:"))
        for i1 in xrange(len(self.lbs)):
            f.write("\t" + "%s" % stat['thres_ary'][i1])
            
        f.write("{0:12}".format("\ntp:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['tp'])
        
        f.write("{0:12}".format("\nfn:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['fn'])

        f.write("{0:12}".format("\nfp:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['fp'])
            
        f.write("{0:12}".format("\ntn:"))
        for lb in self.lbs:
            f.write("\t" + "%d" % stat['event_wise'][lb]['tn'])
            
        f.write("{0:12}".format("\nprecision:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['precision'])
            
        f.write("{0:12}".format("\nrecall:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['recall'])
        
        f.write("{0:12}".format("\nf_value:"))
        for lb in self.lbs:
            f.write("\t" + "%.3f" % stat['event_wise'][lb]['f_value'])
    
        f.write("\n\n# --------- Total evaluation (Mix all events together) ---------")
        f.write("{0:16}".format("\ntotal tp:") + "\t%d" % stat['overall']['tp'])
        f.write("{0:16}".format("\ntotal fn:") + "\t%d" % stat['overall']['fn'])
        f.write("{0:16}".format("\ntotal fp:") + "\t%d" % stat['overall']['fp'])
        f.write("{0:16}".format("\ntotal tn:") + "\t%d" % stat['overall']['tn'])
        f.write("\n")
        f.write("{0:18}".format("\ntotal precision:") + "\t%.3f" % stat['overall']['precision'])
        f.write("{0:18}".format("\ntotal recall:") + "\t%.3f" % stat['overall']['recall'])
        f.write("{0:18}".format("\ntotal f_value:") + "\t%.3f" % stat['overall']['f_value'])
        f.write("{0:18}".format("\ntotal error_rate:") + "\t%.3f" % stat['overall']['error_rate'])
        f.write("\n")
        f.close()
        print( "Write out stat to", stat_path, "successfully!")
        
    def print_stat(self, stat_path):
        fr = open(stat_path, 'r')
        print(fr.read())
        fr.close()
        
    def _reorder_mat_list(self, pd_mat_list, pd_na_list, gt_na_list):
        indexes = [pd_na_list.index(e) for e in gt_na_list]
        return [pd_mat_list[idx] for idx in indexes]

    def _get_digit_mat_from_thres_ary(self, pd_prob_mat, thres_ary):
        pd_digit_mat = np.zeros_like(pd_prob_mat)
        for k in xrange(len(self.lbs)):
            pd_digit_mat[:, k][np.where(pd_prob_mat[:, k] > thres_ary[k])] = 1
        return pd_digit_mat
        
    def _get_stats_from_digit_mat(self, pd_digit_mat, gt_mat):
        stat = {}

        # overall stat
        (tp, fn, fp, tn) = eva.tp_fn_fp_tn(pd_digit_mat, gt_mat, 0.5)
        error_rate = eva.error_rate(pd_digit_mat, gt_mat)
        stat['overall'] = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'error_rate': error_rate}

        # element-wise stat
        stat['event_wise'] = {}
        for k in xrange(len(self.lbs)):
            (tp, fn, fp, tn) = eva.tp_fn_fp_tn(pd_digit_mat[:, k], gt_mat[:, k], 0.5)
            stat['event_wise'][self.lbs[k]] = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
            
        return stat
        
    def  _precision(self, tp, fn, fp, tn):
        if (tp + fp) == 0: 
            return 0
        else:
            return float(tp) / (tp + fp)
            
    def _recall(self, tp, fn, fp, tn):
        if (tp + fn) == 0:
            return 0
        else:
            return float(tp) / (tp + fn)