from typing import Union
import numpy as np
from numpy.random import Generator, PCG64


class Env(object):
    def __init__(self, K: int, d: int, f_real: callable, random_seed: int = 12345) -> None:
        """The contextual bandits environment

        Args:
            K (int): Number of arms.
            d (int): Number of dimension.
            f_real (callable): The real function that determines the mean reward.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        self.K = K
        self.d = d
        self.f_real = f_real
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.context_ = []
        self.t = 1

    def deal(self):
        context = self.random_generator.uniform(low=0.0, high=1.0, size=self.d)
        self.context_.append(context)
        return context

    def response(self, action):
        assert action >= 0 and action <= self.K - 1, "the action is out of bound"
        reward = self.f_real(self.context_[-1], action) + self.random_generator.normal(loc=0.0, scale=0.5)
        return reward
