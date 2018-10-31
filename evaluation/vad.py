"""
SUMMARY:  Event detection using threshold, return list of <bgn, fin>
--------------------------------------
"""
import numpy as np


def activity_detection(x, thres, n_smooth=1, n_salt=0):
    """Activity detection. 
    
    Args:
      x: array
      thres:    float, threshold
      n_smooth: integar, number of frames to smooth. 
      n_salt:   integar, number of frames equal or shorter this value will be 
                deleted. Set this value to 0 means do not use delete_salt_noise. 
    
    Return: list of <bgn, fin>
    """
    locts = np.where(x>thres)[0]
    locts = smooth(locts, n_smooth)
    locts = delete_salt_noise(locts, n_salt)
    lists = find_bgn_fin_pairs(locts)
    return lists


def smooth(locts, n_smooth):
    """Smooth the loctation array
    
    Args:
      locts: int array, location array
      n_smooth: integar, number of points to smooth. Set to 0 means do not use
                this function. 
    
    Return:
      locts: smoothed location array
      
    Eg.
    input: np.array([3,4,7,8])
    return: np.array([3,4,5,6,7,8])
    """
    if len(locts)==0:
        return locts
    else:
        smooth_locts = [locts[0]]
        for i1 in xrange(1,len(locts)):
            if locts[i1]-locts[i1-1] <= n_smooth:                
                for i2 in xrange(locts[i1-1]+1, locts[i1]):
                    smooth_locts.append(i2)
            smooth_locts.append(locts[i1])
        return smooth_locts
    
    
def delete_salt_noise(locts, n_salt):
    """Delete salt noise. 
    
    Args:
      locts:  int array. 
      n_salt: integar, number of frames equal or shorter this value will be 
              deleted. Set this value to 0 means do not use delete_salt_noise. 
              
    Return:
      locts: int array. 
    """

    if len(locts) == 0:
        return locts
    else:
        # Pseudo bgn and fin for convenience. 
        locts = [-n_salt - 10] + locts + [n_salt + locts[-1] + 10]
        locts_to_delete = []
        pt = 0
        cnt = 1
        # print locts
        for i1 in xrange(1, len(locts)):
            if locts[i1] - locts[i1 - 1] == 1:
                cnt += 1
            else:
                cnt += 1
                if cnt <= n_salt:
                    locts_to_delete += range(pt, i1)
                pt = i1
                cnt = 0
    
        locts_to_delete += [len(locts) - 1]
        locts = np.delete(locts, locts_to_delete)
        return locts
        
        
def find_bgn_fin_pairs(locts):
    """Find pairs of <bgn, fin> from loctation array
    """
    if len(locts)==0:
        return []
    else:
        bgns = [locts[0]]
        fins = []
        for i1 in xrange(1,len(locts)):
            if locts[i1]-locts[i1-1]>1:
                fins.append(locts[i1-1])
                bgns.append(locts[i1])
        fins.append(locts[-1])
            
    assert len(bgns)==len(fins)
    lists = []
    for i1 in xrange(len(bgns)):
        # lists.append({'bgn':bgns[i1], 'fin':fins[i1]})
        lists.append([bgns[i1], fins[i1]])
    return lists
    