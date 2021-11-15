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
    # A is the d*1 matrix,
    
    # this function return the reward,
    assert len(context.shape) == 1, "GetRealReward: contex is not a vector" 
    assert len(context.shape) == 1, "GetRealReward: A is not a vector" 
    assert len(context) == len(A), "GetRealReward: length of A not equal to length of context" 
        
    
    # return 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct)) + xi, xi follows standard normal distribution
    innerproduct = A.dot(context)
    # return 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct)) - 1 + np.random.normal(loc=0, scale=1)
    return 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct)) - 1
