# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:22:00 2021

@author: lzt68
"""
# following are functions that are used to play the game
import numpy as np


def SampleContext(d: int, K: int) -> np.ndarray:
    """This function return context, as an K*d matrix, each row represents a context of action

    Args:
        d (int): Dimension of context
        K (int): Number of arms

    Returns:
        context: an np.ndarray whose shape is (K, d), each row represents a context
    """
    context = np.random.normal(loc=0, scale=1, size=(K, d // 2))
    length = np.sqrt(np.sum(context * context, axis=1, keepdims=True))
    context = np.tile(context, (1, 2))
    length = np.tile(length, (1, d))
    context = context / length / np.sqrt(2)  # each column represent a context
    return context


def GetRealReward(context: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Given the context, return the realized reward

    Args:
        context (np.ndarray): An np.ndarray whose shape is (K, d), each column represents a context of an arm
        A (np.ndarray): The parameter of this reward function

    Returns:
        reward: an np.ndarray whose shape is (K,), reward = context^T A^T A context + N(0, 0.05^2)
    """
    if len(context.shape) == 1:
        return context.transpose().dot(A.transpose().dot(A)).dot(context) + np.random.normal(loc=0, scale=0.05)
        # return context.transpose().dot(A.transpose().dot(A)).dot(context)
    else:
        return np.diag(context.dot(A.transpose().dot(A)).dot(context.transpose())) + np.random.normal(loc=0, scale=0.05, size=context.shape[0])
        # return np.diag(context.dot(A.transpose().dot(A)).dot(context.transpose()))
