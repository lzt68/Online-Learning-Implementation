import numpy as np
from random import Random
from typing import Union
from typing import Tuple
from numpy.random import Generator, PCG64


class Env_Gaussian_Fixedmu0(object):
    def __init__(
        self,
        K: int,
        n: int,
        mu0: Union[np.float64, int, float],
        r_list: np.ndarray,
        sigma: float = 1.0,
        random_seed: int = 12345,
        alpha: float = 1 / 6,
    ) -> None:
        """The environment that return Gaussian reward with fixed variance.

        Args:
            K (int): Number of alternative arms.
            n (int): Total number of rounds
            mu0 (Union[np.float64, int, float]): The mean reward of default value
            r_list (np.ndarray): Mean reward of each arm.
            sigma (float, optional): The standard deviation of Normal distribution. Defaults to 1.0.
            random_seed (int, optional): Random Seed. Defaults to 12345.
            alpha (float, optional): The factor of safety threshold
        """
        assert r_list.shape[0] == K, "number of arms doesn't match"
        assert mu0 > 0.0 and mu0 < 1.0, "mu0 is not in (0, 1)"
        assert alpha >= 0.0 and alpha <= 1.0, "mu0 is not in [0, 1]"

        self.K = K
        self.n = n
        self.mu0 = mu0
        self.alpha = alpha
        self.r_list = r_list
        self.sigma = sigma
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.violation = False  # Mark whether there is violation of safety constraint
        self.total_reward = 0

        self.t = 0

    def response(self, arm):
        if arm == 0:
            reward = self.mu0
            self.total_reward += self.mu0
        else:
            reward = self.random_generator.normal(loc=self.r_list[arm - 1], scale=self.sigma)
            self.total_reward += self.r_list[arm - 1]

        if self.total_reward < self.t * (1 - self.alpha) * self.mu0:
            # we wiolate the safety constraint
            self.if_violation = True

        self.t += 1
        return reward

    def if_stop(self):
        return self.t >= self.n

    def if_violation(self):
        return self.violation
