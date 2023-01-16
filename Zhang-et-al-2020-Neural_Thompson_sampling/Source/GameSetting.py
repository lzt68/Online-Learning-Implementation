# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:22:00 2021

@author: lzt68
"""
# following are functions that are used to play the game 
import numpy as np

def SampleContext(d, K):
    # according to the description, the context is uniformly distributed on a d-dimension sphere
    # d is the dimension of context, a scalar, and it is required to be an even number
    # K is the total number of arts, a scalar
    
    # this function return context, as an d*K matrix, each column corresponds a context of action
    
    context = np.random.normal(loc=0, scale=1, size=(np.int64(d / 2), K))
    length = np.sqrt( np.sum(context * context, axis = 0) )
    context = np.tile(context, (2, 1))
    length = np.tile(length, (d, 1))
    context = context / length / np.sqrt(2) # each column represent a context
    return context

def GetRealReward(context, A):
    # context is the context of arm, a d*1 vector
    # A is the d*1 matrix,
    
    # return 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct)) + xi, xi follows standard normal distribution
    innerproduct = A.dot(context)
    return 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct)) - 1 + np.random.normal(loc = 0, scale = 0.05)
    # return 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct)) - 1

# unit test
# arm = SampleContext(d = 4, K = 2)
# print(np.sum(arm[0:4, 0] * arm[0:4, 0]))
# print(np.sum(arm[0:4, 1] * arm[0:4, 1]))

# A = np.array([1, 2, 3])
# context = np.array([[1, 0], [0, 1], [1, 1]])
# print(GetRealReward(context, A))
