# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:22:00 2021

@author: lzt68
"""
# following are functions that are used to play the game 
import numpy as np

def SampleContext(d, K):
    # according to the description, the context is uniformly distributed on a d-dimension sphere
    # d is the dimension of context, a scalar
    # K is the total number of arts, a scalar
    
    # this function return context, as an d*K matrix, each column corresponds a context of action
    
    context = np.random.normal(loc=0, scale=1, size=(d, K))
    length = np.sqrt( np.sum(context * context, axis = 0) )
    length = np.tile(length, (d, 1))
    context = context / length # each column represent a context
    return context

def GetRealReward(context, A):
    # context is the context of arm, a d*1 vector
    # A is the d*d matrix,
    
    # this function return the reward
    
    # return context.transpose().dot(A.transpose().dot(A)).dot(context) + np.random.normal(loc=0, scale=1)
    if len(context.shape) == 1:
        return context.transpose().dot(A.transpose().dot(A)).dot(context) + np.random.normal(loc = 0, scale = 1)
    else:
        return np.diag(context.transpose().dot(A.transpose().dot(A)).dot(context)) + np.random.normal(loc=0, scale=1, size = context.shape[1])