import numpy as np
from numpy.random import Generator, PCG64


class H1_OnlineLinearEnv(object):
    def __init__(
        self,
        m: int,
        T: int,
        d: int,
        # b: np.ndarray,
        random_seed: int = 12345,
    ) -> None:
        """The env that reproduces the setting in section H.1

        Args:
            m (int): Number of resources.
            T (int): Number of rounds.
            d (int): Dimensions of decision variables.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        self.m = m
        self.T = T
        self.d = d
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.theta = self.random_generator.normal(loc=0.0, scale=1.0, size=m)
        self.theta = self.theta / np.linalg.norm(x=self.theta, ord=2)
        self.alpha = self.random_generator.beta(a=1.0, b=3.0)
        self.beta = self.random_generator.uniform(low=0.25, high=0.75)

        self.p = self.random_generator.binomial(n=1, p=(1 + self.alpha) / 2, size=m)
        self.rho = self.p * self.beta
        self.b = self.T * self.rho

        self.total_consumption_ = np.zeros(m)
        self.stop = False

        self.c = np.zeros((m, d, T))
        self.r = np.zeros(T)

        self.t = 1

    def deal(self):
        if self.t <= self.T and (not self.stop):
            c_j = np.zeros((self.m, self.d))
            for ii in range(self.m):
                c_j[ii, :] = self.random_generator.binomial(n=1, p=self.p[ii], size=self.d)

            r_j = self.theta @ c_j + self.random_generator.normal(loc=0.0, scale=1.0)
            r_j[r_j > 10.0] = 10

            self.c[:, :, self.t - 1] = c_j
            self.r[self.t - 1] = r_j
            return r_j, c_j
        else:
            return None

    def observe(self, action):
        consumption = self.c[:, :, self.t - 1] @ action
        self.total_consumption_ += consumption
        self.t += 1
        if self.t > self.n or np.any(self.total_consumption_ > self.b):
            self.stop = True

    def if_stop(self):
        return self.stop
