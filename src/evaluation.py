"""
import numpy as np

def score(t, y, eps=1e-6):
    '''
    Inputs:
    - t: A numpy array, representing the targets
    - y: A numpy array, representing the predictions 
    
    For detailed explenation please visit the following links:
    - https://www.kaggle.com/c/two-sigma-financial-modeling#evaluation
    - https://en.wikipedia.org/wiki/Coefficient_of_determination
    '''
    target_mean = np.mean(t)
    total_sum_of_squares  = np.sum(np.power(t-target_mean, 2))
    explained_sum_of_squares = np.sum(np.power(t-y, 2))
    coefficient_of_determination = 1 - explained_sum_of_squares/(total_sum_of_squares + eps)
    R = np.sign(coefficient_of_determination) * np.sqrt(np.absolute(coefficient_of_determination))
    return R
"""

import numpy as np
from sklearn.metrics import r2_score

def score(t, y):
    coefficient_of_determination = r2_score(t, y)
    R = np.sign(coefficient_of_determination) * np.sqrt(np.absolute(coefficient_of_determination))
    return R