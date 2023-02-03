import numpy as np
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

    def action(self):
        pass

    def update(self):
        pass
