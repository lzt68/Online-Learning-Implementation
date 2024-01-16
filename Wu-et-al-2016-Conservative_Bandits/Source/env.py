import numpy as np
from random import Random
from typing import Union
from typing import Tuple
from numpy.random import Generator, PCG64


class Env_Gaussian_Fixedmu0(object):
    def __init__(
        self,
        K: int,
        mu0: Union[np.float64, int, float],
        r_list: np.ndarray,
        sigma: float = 1.0,
        random_seed: int = 12345,
    ) -> None:
        """The environment that return Gaussian reward with fixed variance.

        Args:
            K (int): Number of alternative arms.
            r_list (np.ndarray): Mean reward of each arm.
            sigma (float, optional): _description_. Defaults to 1.0.
            random_seed (int, optional): _description_. Defaults to 12345.
        """
        assert r_list.shape[0] == K, "number of arms doesn't match"
        assert mu0 > 0.0 and mu0 < 1.0, "mu0 is not in (0, 1)"

        self.K = K
        self.mu0 = mu0
        self.r_list = r_list
        self.sigma = sigma
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

    def response(self, arm):
        if arm == 0:
            reward = self.mu0
        else:
            reward = self.random_generator.normal(loc=self.r_list[arm], scale=self.sigma)
        return reward
