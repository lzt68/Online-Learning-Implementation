import numpy as np
from numpy.random import Generator, PCG64


class Environment_Gaussian:
    # the demand follows Gaussian Distribution
    def __init__(self, rlist: np.array, K: int, random_seed=12345):
        """The environment that return gaussian reward

        Args:
            rlist (np.array): The mean reward of each arm
            K (int): The number of arms
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert len(rlist) == K, "number of arms doesn't match"

        self.rlist = rlist
        self.K = K
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

    def response(self, arm):
        # vectorize binomial sample function to accelerate
        reward = self.random_generator.normal(loc=0.0, scale=0.5) + self.rlist[arm - 1]
        return reward
