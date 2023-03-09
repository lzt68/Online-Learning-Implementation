import numpy as np
from numpy.random import Generator, PCG64
from typing import Union


class Env(object):
    def __init__(self, m: int, n: int, b: np.ndarray, pi: Union[None, np.ndarray], a: Union[None, np.ndarray], random_seed: int = 12345) -> None:
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        if pi is None:
            self.pi = self.random_generator.uniform(low=0.0, high=1.0, size=(n))
        else:
            assert pi.shape[0] == n and len(pi.shape) == 1, "Number of rounds doesn't match"
            self.pi = pi

        if a is None:
            self.a = self.random_generator.uniform(low=0.0, high=1.0, size=(m, n))
        else:
            assert a.shape[0] == m and a.shape[1] == n and len(a.shape) == 2, "Size of consumption matrix doesn't match"
            self.a = a

        self.t = 1

        # shuffle the arm
        index = np.arange(n)
        self.random_generator.shuffle(index)
        self.a = self.a[:, index]
        self.pi = self.pi[index]

    def deal(self):
        if self.t <= self.n:
            pi_t = self.pi[self.t - 1]
            a_t = self.a[:, self.t - 1]
            self.t += 1
            return pi_t, a_t
        else:
            return None

    def if_stop(self):
        return self.t > self.n
