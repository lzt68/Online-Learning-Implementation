from typing import Union, Tuple
import numpy as np
from numpy.random import Generator, PCG64


class Env_FixedActionSpace(object):
    def __init__(self, theta: Union[None, np.ndarray] = None, d: int = 2, random_seed: int = 12345, R: float = 0.1) -> None:
        """The environment that always adopts fixed action space, which is a unit ball in the d-dimension space

        Args:
            theta (np.ndarray): The unknown theta of the environment
            d (int, optional): Dimension of the action space. Defaults to 2.
            random_seed (int, optional): The random seed. Defaults to 12345.
            R (float, optional): The noise is R-subgaussian random variable
        """
        self.d = d
        self.random_seed = random_seed
        self.R = R
        self.random_generator = Generator(PCG64(random_seed))
        if theta is not None:
            assert theta.shape[0] == d, "The size of theta doesn't match"
            self.theta = theta
        else:
            self.theta = self.random_generator.uniform(low=0.0, high=1.0, size=2)
            self.theta = self.theta / np.linalg.norm(self.theta) * self.random_generator.uniform(low=0.0, high=1.0)

    def response(self, action: np.ndarray) -> Tuple[np.float64, np.float64]:
        """Given the action of the agent, generate the reward

        Args:
            action (np.ndarray): the vector of action

        Returns:
            reward_mean: the mean reward of this action, <action, self.theta>
            reward_real: the mean reward of this action + the realized noise
        """

        assert action.shape[0] == self.d and len(action.shape) == 1, "The action doesn't match the dimension"
        # gap = np.linalg.norm(action-self.theta)
        # print(f"gap {gap}")

        reward_mean = action @ self.theta
        reward_real = action @ self.theta + self.random_generator.normal(loc=0.0, scale=self.R)
        return reward_mean, reward_real
