import numpy as np
from numpy import linalg
from copy import deepcopy


class ConfidenceBall_Agent(object):
    def __init__(self) -> None:
        pass


class OFUL_Agent(object):
    def __init__(
        self,
        d: int,
        R: float,
        lambda_: float,
        L: float,
        S: float,
    ) -> None:
        """Initialization of OFUL algorithm,
        O: Optimism, U: Uncertainty, F: Face, L: Linear
        We assume the action space is always the unit ball in d-dimension space

        Args:
            d (int): The dimension of the decision space
            R (float): The nois is conditionally R-subgaussian random variable
            _lambda (float): Regularization coefficient
            L (float): \|X_t\|_2 \le L
            S (float): \|\theta_*\|_2 \le S
        """
        self.d = d
        self.R = R
        self.lambda_ = lambda_
        self.L = L
        self.S = S

        self.V = np.eye(d) * lambda_
        self.H_action = list()  # history of reward
        self.H_reward = list()  # history of action

        self.t = 0

    def action(self):
        theta = self.get_theta()
        theta_l2 = linalg.norm(theta, ord=2)
        if np.abs(theta_l2) < 1e-4:
            act = np.zeros(self.d)
        else:
            act = theta / theta_l2
        return act

    def update(self, reward):
        self.t += 1

    def get_theta(self):
        # solve the problem
        # \tilde{\theta}_t=\arg\max_{\theta\in C_{t-1}}\|\theta\|_2
        return np.zeros(self.d)
