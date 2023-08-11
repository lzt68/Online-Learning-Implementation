from typing import Union

import numpy as np
from numpy import linalg
from copy import deepcopy
from scipy.optimize import minimize


def Get_w_star(mu: np.ndarray):
    """Given the array of mean reward, solve the optimization problem and get the optimal pulling fraction
    The optimization target is
    $$
        w^*(\mu)=\arg\max_{w\in\Sigma_K} \min_{a\ne 1}(w_1+w_a)I_{\frac{w_1}{w_1+w_a}}(\mu_1, \mu_a)
    $$
    where $I_{\alpha}(\mu_1, \mu_2)=\alpha d\left(\mu_1, \alpha\mu_1+(1-\alpha)\mu_2\right)+(1-\alpha)d(\mu_2, \alpha\mu_1+(1-\alpha)\mu_2)$
    and $d(x,y)=x\log\frac{x}{y}+(1-x)\log\frac{1-x}{1-y}$

    Args:
        mu (np.ndarray): Array of mean rewards
    """


class C_Tracking(object):
    def __init__(self, K: int = 4, delta: Union[float, np.float64] = 0.1) -> None:
        """Adopt C-Tracking rule to pull arms

        Args:
            K (int, optional): Number of arms. Defaults to 4.
            delta (Union[float, np.float64], optional): Threshold of failure probability. Defaults to 0.1.
        """

        self.K = K
        self.delta = delta

        # generate dictionary to record realized reward and consumption
        self.action_ = list()
        self.reward_ = dict()
        self.consumption_ = dict()
        for kk in range(1, K + 1):
            self.reward_[kk] = list()
            self.consumption_[kk] = list()

        self.w = list()  # record the optimized pulling fraction in each round

    def action(self):
        pass

    def observe(self, r, d):
        act = self.action_[-1]
        self.reward_[act].append(r)
        self.consumption_[act].append(d)
