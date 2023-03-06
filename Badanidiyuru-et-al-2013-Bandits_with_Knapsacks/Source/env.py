import numpy as np
from random import Random
from typing import Tuple
from numpy.random import Generator, PCG64


class Env_FixedConsumption:
    def __init__(self, r_list: np.ndarray = np.array([0.5, 0.25]), d_list: np.ndarray = np.array([[0.1, 0.1]]), m: int = 2, B: float = 10.0, d: int = 1, random_seed=12345) -> None:
        """In this environment, the reward is stochastic, the consumption is fixed

        Args:
            r_list (np.ndarray, optional): The mean reward of each arm. Defaults to np.array([0.5, 0.25]).
            d_list (np.ndarray, optional): The mean consumption of each arm. Defaults to np.array([[0.1, 0.1]]).
            m (int, optional): The total number of arms. Defaults to 2.
            B (int, optional): Initial Resource. We assume all the resources share the same initial value. Defaults to 10.
            d (int, optional): Number of resource. Defaults to 1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert d_list.shape[0] == d, "number of resources doesn't match"
        assert r_list.shape[0] == d_list.shape[1], "number of arms doesn't match"
        assert r_list.shape[0] == m, "number of arms doesn't match"

        self.r_list = r_list
        self.d_list = d_list
        self.m = m
        self.B = B
        self.d = d
        self.consumption = np.zeros(d)
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

    def response(self, arm: int) -> Tuple[np.ndarray, float]:
        """Respond to the selected arm of agent

        Args:
            arm (int): arm index range from 1 to K

        Returns:
            consumption (np.ndarray): The consumption of each resources
            reward (float): The realized reward
        """
        if not self.stop:
            consumption = self.d_list[:, arm - 1]
            reward = self.random_generator.binomial(n=1, p=self.r_list[arm - 1])
            self.consumption += consumption
            if np.any(self.consumption >= self.B):
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_Uncorrelated_Reward:
    def __init__(self, r_list: np.ndarray = np.array([0.5, 0.25]), d_list: np.ndarray = np.array([[0.1, 0.1]]), m: int = 2, B: float = 10.0, d: int = 1, random_seed=12345) -> None:
        """In this environment, the reward and demand are independent

        Args:
            r_list (np.ndarray, optional): The mean reward of each arm. Defaults to np.array([0.5, 0.25]).
            d_list (np.ndarray, optional): The mean consumption of each arm. Defaults to np.array([[0.1, 0.1]]).
            m (int, optional): The total number of arms. Defaults to 2.
            B (int, optional): Initial Resource. We assume all the resources share the same initial value. Defaults to 10.
            d (int, optional): Number of resource. Defaults to 1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert d_list.shape[0] == d, "number of resources doesn't match"
        assert r_list.shape[0] == d_list.shape[1], "number of arms doesn't match"
        assert r_list.shape[0] == m, "number of arms doesn't match"

        self.r_list = r_list
        self.d_list = d_list
        self.m = m
        self.B = B
        self.d = d
        self.consumption = np.zeros(d)
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

    def response(self, arm: int) -> Tuple[np.ndarray, float]:
        """Respond to the selected arm of agent

        Args:
            arm (int): arm index range from 1 to K

        Returns:
            consumption (np.ndarray): The consumption of each resources
            reward (float): The realized reward
        """
        if not self.stop:
            consumption = np.zeros(self.d)
            for ll in range(self.d):
                consumption[ll] = self.random_generator.uniform(low=0.0, high=1.0) <= self.d_list[ll, arm - 1]
            consumption = consumption.astype(float)
            reward = self.random_generator.uniform(low=0.0, high=1.0) <= self.r_list[arm - 1]
            reward = reward.astype(float)
            self.consumption += consumption
            if np.any(self.consumption >= self.B):
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_Correlated_Uniform:
    def __init__(self, r_list: np.ndarray = np.array([0.5, 0.25]), d_list: np.ndarray = np.array([[0.1, 0.1]]), m: int = 2, B: float = 10.0, d: int = 1, random_seed=12345) -> None:
        """In this environment, the reward and demand are dependent
        reward = \mathbb{1}(U <= r), consumption = \mathbb{1}(U <= d),
        where U follows U(0, 1)

        Args:
            r_list (np.ndarray, optional): The mean reward of each arm. Defaults to np.array([0.5, 0.25]).
            d_list (np.ndarray, optional): The mean consumption of each arm. Defaults to np.array([[0.1, 0.1]]).
            m (int, optional): The total number of arms. Defaults to 2.
            B (int, optional): Initial Resource. We assume all the resources share the same initial value. Defaults to 10.
            d (int, optional): Number of resource. Defaults to 1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert d_list.shape[0] == d, "number of resources doesn't match"
        assert r_list.shape[0] == d_list.shape[1], "number of arms doesn't match"
        assert r_list.shape[0] == m, "number of arms doesn't match"

        self.r_list = r_list
        self.d_list = d_list
        self.m = m
        self.B = B
        self.d = d
        self.consumption = np.zeros(d)
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

    def response(self, arm: int) -> Tuple[np.ndarray, float]:
        """Respond to the selected arm of agent

        Args:
            arm (int): arm index range from 1 to K

        Returns:
            consumption (np.ndarray): The consumption of each resources
            reward (float): The realized reward
        """
        if not self.stop:
            U = self.random_generator.uniform(low=0.0, high=1.0)
            consumption = (U <= self.d_list[:, arm - 1]).astype(float)
            reward = float(U <= self.r_list[arm - 1])
            self.consumption += consumption
            if np.any(self.consumption >= self.B):
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop
