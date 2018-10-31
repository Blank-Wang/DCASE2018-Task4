"""
SUMMARY:  Metrics
--------------------------------------
"""
import numpy as np
from sklearn import metrics

def eer(pred, gt):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, drop_intermediate=True)
    
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]
        
    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)  
    return EER
    
def tp_fn_fp_tn(pred, gt, thres):
    pred_digt = np.zeros_like(pred)
    pred_digt[np.where(pred > thres)] = 1.
    tp = np.sum(pred_digt + gt > 1.5)
    fn = np.sum(gt - pred > 0.5)
    fp = np.sum(pred_digt - gt > 0.5)
    tn = np.sum(pred_digt + gt < 0.5)
    return tp, fn, fp, tn
    
def precision(pred, gt, thres):
    (tp, fn, fp, tn) = tp_fn_fp_tn(pred, gt, thres)
    if (tp + fp) == 0: 
        return 0
    else:
        return float(tp) / (tp + fp)
    
def recall(pred, gt, thres):
    (tp, fn, fp, tn) = tp_fn_fp_tn(pred, gt, thres)
    if (tp + fn) == 0:
        return 0
    else:
        return float(tp) / (tp + fn)
    
def f_value(prec, rec):
    if (prec + rec) == 0:
        return 0
    else:
        return 2 * prec * rec / (prec + rec)

def roc_auc(pred, gt):
    return metrics.roc_auc_score(gt, pred)
    
def error_rate(pred, gt):
    """Ref: Mesaros, Annamaria, Toni Heittola, and Tuomas Virtanen. "Metrics for 
       polyphonic sound event detection." Applied Sciences 6.6 (2016): 162.
    """
    er_ary = []
    for n in xrange(len(pred)):
        (tp, fn, fp, tn) = tp_fn_fp_tn(pred[n], gt[n], thres=0.5)
        n_substitue = min(fn, fp)
        n_delete = max(0, fn - fp)
        n_insert = max(0, fp - fn)
        n_gt = np.sum(gt[n])
        if n_gt == 0:
            er = 0.
        else:
            er = (n_substitue + n_delete + n_insert) / float(n_gt)
        er_ary.append(er)
    return np.mean(np.array(er_ary))
