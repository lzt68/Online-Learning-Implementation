from typing import Union
import numpy as np


class Falcon(object):
    def __init__(self, K: int, n: int, delta: Union[float, np.float64], c: Union[float, np.float64], T: Union[int, None] = None) -> None:
        """The algorithm Falcon assume the size of function class is known to us. Here we denote it as n.
        We will assign the epoch schedule through the parameter T.
        If T is assigned to this algorithm, we will set $\tau_m=2T^{1-2^{-m}}$.
        If T is unknown to us, we will set $\tau_m = 2^m$

        Args:
            K (int): Number of arms.
            n (int): Size of function class.
            delta (Union[float, np.float64]): Confidence level.
            c (Union[float, np.float64]): Tuning parameter
            T (Union[int, None], optional): Number of rounds. Defaults to None.
        """
        self.m = 0  # index of current epoch
        self.t = 1  # index of current round
        self.gamma_m = 0  # the scalar that balance exploration and exploitation

        self.K = K
        self.n = n
        self.delta = delta
        self.c = c
        self.T = T
        T_is_None = T is None

    def action(x_t):
        pass
