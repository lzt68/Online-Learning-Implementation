# source file of environments
import numpy as np


class Env_FixedConsumption:
    def __init__(self, r_list=[0.5, 0.25], d_list=[0.1, 0.1], K=2, C=10, random_seed=12345) -> None:
        """In this environment, the reward is stochastic, the consumption is fixed
        Args:
            r_list (list, optional): The mean reward of each arm. Defaults to [0.5, 0.25].
            d_list (list, optional): The mean consumption of each arm. Defaults to [0.1, 0.1].
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to 10.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert len(r_list) == len(d_list), "number of arms doesn't match"
        assert len(r_list) == K, "number of arms doesn't match"
        assert C > 0, "initial resource should be greater than 0"

        self.r_list = r_list
        self.d_list = d_list
        self.K = K
        self.C = C
        self.consumption = 0
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def response(self, arm):
        if not self.stop:
            consumption = self.d_list[arm - 1]
            reward = np.random.binomial(n=1, p=self.r_list[arm - 1])
            self.consumption += consumption
            if self.consumption >= self.C - 1:
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_Uncorrelated_Reward:
    def __init__(self, r_list=[0.5, 0.25], d_list=[0.1, 0.1], K=2, C=10, random_seed=12345) -> None:
        """In this environment, the reward and demand are independent
        Args:
            r_list (list, optional): The mean reward of each arm. Defaults to [0.5, 0.25].
            d_list (list, optional): The mean consumption of each arm. Defaults to [0.1, 0.1].
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to 10.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert len(r_list) == len(d_list), "number of arms doesn't match"
        assert len(r_list) == K, "number of arms doesn't match"
        assert C > 0, "initial resource should be greater than 0"

        self.r_list = r_list
        self.d_list = d_list
        self.K = K
        self.C = C
        self.consumption = 0
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def response(self, arm):
        if not self.stop:
            consumption = np.random.binomial(n=1, p=self.d_list[arm - 1])
            reward = np.random.binomial(n=1, p=self.r_list[arm - 1])
            self.consumption += consumption
            if self.consumption >= self.C - 1:
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_Correlated_Uniform:
    def __init__(self, r_list=[0.5, 0.25], d_list=[0.1, 0.1], K=2, C=10, random_seed=12345) -> None:
        """In this environment, the reward and demand are dependent
        reward = \mathbb{1}(U <= r), consumption = \mathbb{1}(U <= d),
        where U follows U(0, 1)

        Args:
            r_list (list, optional): The mean reward of each arm. Defaults to [0.5, 0.25].
            d_list (list, optional): The mean consumption of each arm. Defaults to [0.1, 0.1].
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to 10.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert len(r_list) == len(d_list), "number of arms doesn't match"
        assert len(r_list) == K, "number of arms doesn't match"
        assert C > 0, "initial resource should be greater than 0"

        self.r_list = r_list
        self.d_list = d_list
        self.K = K
        self.C = C
        self.consumption = 0
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def response(self, arm):
        if not self.stop:
            U = np.random.uniform(low=0.0, high=1.0)
            consumption = U <= self.d_list[arm - 1]
            reward = U <= self.r_list[arm - 1]
            self.consumption += consumption
            if self.consumption >= self.C - 1:
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class Env_FixedConsumption_Gaussian:
    def __init__(self, r_list=[0.5, 0.25], d_list=[0.1, 0.1], K=2, C=10, random_seed=12345) -> None:
        """In this environment, the reward and demand are independent
        Args:
            r_list (list, optional): The mean reward of each arm. Defaults to [0.5, 0.25].
            d_list (list, optional): The mean consumption of each arm. Defaults to [0.1, 0.1].
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to 10.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert len(r_list) == len(d_list), "number of arms doesn't match"
        assert len(r_list) == K, "number of arms doesn't match"
        assert C > 0, "initial resource should be greater than 0"

        self.r_list = r_list
        self.d_list = d_list
        self.K = K
        self.C = C
        self.consumption = 0
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def response(self, arm):
        if not self.stop:
            consumption = self.d_list[arm - 1]
            reward = np.random.normal(loc=0.0, scale=0.5) + self.r_list[arm - 1]  # variance = 0.25
            self.consumption += consumption
            if self.consumption >= self.C - 1:
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop
