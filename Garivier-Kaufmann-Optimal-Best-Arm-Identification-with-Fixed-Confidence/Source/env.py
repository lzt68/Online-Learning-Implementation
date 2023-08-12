from typing import Union, Tuple
import numpy as np
from numpy.random import Generator, PCG64


class Env__Deterministic_Consumption(object):
    def __init__(
        self,
        K: int = 4,
        d: np.ndarray = np.ones(4),
        r: np.ndarray = np.array([0.5, 0.45, 0.43, 0.4]),
        random_seed: int = 12345,
    ) -> None:
        """Pulling each arm will consume 1 unit of resources

        Args:
            K (int, optional): Number of arms. Defaults to 4.
            d (np.ndarray, optional): Deterministic consumption of each arm. Defaults to np.ones(4).
            r (np.ndarray, optional): Mean reward of pulling arms. Defaults to np.array([0.5, 0.45, 0.43, 0.4]).
            random_seed (int, optional): The random seed.. Defaults to 12345.
        """
        assert len(d.shape) == 1 and d.shape[0] == K, "The dimension of d doesn't match"
        assert len(r.shape) == 1 and r.shape[0] == K, "The dimension of r doesn't match"
        assert np.all(r <= 1) and np.all(r >= 0), "The mean reward should be in [0, 1]"

        self.K = K
        self.d = d
        self.r = r
        self.t = 1
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

    def response(self, action: int) -> Tuple[np.float64, np.float64]:
        """Given the pulling arm, return the realized reward and consumption

        Args:
            action (int): Arm index, an integer in [K]

        Returns:
            reward: The realized reward
            consumption: The realized consumption
        """
        assert action >= 1 and action <= self.K, "The arm index should be in [K]"

        consumption = self.d[action - 1]
        reward = self.random_generator.binomial(1, p=self.r[action - 1])
        self.t += 1
        return reward, consumption


# %% unit test 1
# np.random.seed(12345)
# T = 10
# K = 4
# action = np.random.randint(low=1, high=K + 1, size=T)

# env = Env__Deterministic_Consumption(K=K)
# for nn in range(10):
#     r, d = env.response(action=action[nn])
#     print(f"round index {nn+1}, pulling arms {action[nn]}; reward {r}, consumption {d}")
