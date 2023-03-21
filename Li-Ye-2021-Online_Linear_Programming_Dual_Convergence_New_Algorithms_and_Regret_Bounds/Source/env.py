import numpy as np
from numpy.random import Generator, PCG64
from typing import Union


class RandomInputI(object):
    def __init__(self, m: int, n: int, b: np.ndarray, random_seed: int = 12345) -> None:
        """The consumption vector a_j ~ i.i.d Uniform(-0.5, 1), the r_j ~ Uniform(0, 10)

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.ndarray): Initial budget
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.total_consumption_ = np.zeros(m)
        self.stop = False

        self.a = np.zeros((m, n))
        self.r = np.zeros(n)

        self.t = 1

    def deal(self):
        if self.t <= self.n and (not self.stop):
            r_j = self.random_generator.uniform(low=0.0, high=10.0)
            a_j = self.random_generator.uniform(low=-0.5, high=1.0, size=self.m)

            self.a[:, self.t - 1] = a_j
            self.r[self.t - 1] = r_j
            return r_j, a_j
        else:
            return None

    def observe(self, action):
        consumption = action * self.a[:, self.t - 1]
        self.total_consumption_ += consumption
        self.t += 1
        if self.t > self.n:
            self.stop = True

    def if_stop(self):
        return self.stop


class RandomInputII(object):
    def __init__(self, m: int, n: int, b: np.ndarray, random_seed: int = 12345) -> None:
        """The consumption vector a_j ~ i.i.d N(0.5, 1), the r_j = \sum_{i=1}^m a_{ij}

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.ndarray): Initial budget
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.total_consumption_ = np.zeros(m)
        self.stop = False

        self.t = 1

    def deal(self):
        if self.t <= self.n and (not self.stop):
            a_j = self.random_generator.normal(loc=0.5, scale=1.0, size=self.m)
            r_j = np.sum(a_j)
            return r_j, a_j
        else:
            return None

    def observe(self, action):
        consumption = action * self.a[:, self.t - 1]
        self.total_consumption_ += consumption
        # if np.any(self.total_consumption_ > self.b):
        #     self.stop = True
        self.t += 1
        if self.t > self.n:
            self.stop = True

    def if_stop(self):
        return self.stop
