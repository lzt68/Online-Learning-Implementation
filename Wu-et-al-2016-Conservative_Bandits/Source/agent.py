from typing import Union
import numpy as np


class Conservative_UCB(object):
    def __init__(
        self,
        K: int,
        mu0: Union[np.float64, int, float],
        delta: Union[np.float64, int, float],
        alpha: Union[np.float64, int, float],
    ) -> None:
        """Implement the Conservative UCB algorithm

        Args:
            K (int): Number of alternative arms
            mu0 (Union[np.float64, int, float]): Mean reward of the default arm.
            delta (Union[np.float64, int, float]): Probability threhsold for the incorrect prediction or the violation.
            alpha (Union[np.float64, int, float]): Safety threshold.
        """
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        assert alpha > 0.0 and alpha < 1.0, "delta is not in (0, 1)"
        assert mu0 > 0.0 and mu0 < 1.0, "mu0 is not in (0, 1)"

        self.K = K
        self.delta = delta
        self.alpha = alpha
        self.mu0 = mu0

        self.t = 1
        self.action_ = list()
        self.total_reward_ = np.zeros(K + 1)  # arm=0, 1, 2, ..., K
        self.pulling_times_ = np.zeros(K + 1)
        self.mean_reward_ = np.zeros(K + 1)

    def action(self):
        # calculate the length of confidence interval
        Delta_t = self.rad_1(self.pulling_times_[1:])

        # calculate the lower bound for all the arms
        lambda_t = self.zeros(self.K + 1)
        lambda_t[0] = self.mu0
        lambda_t[1:] = self.mean_reward_[1:] - Delta_t
        lambda_t[1:] = np.maximum(lambda_t[1:], 0)

        # calculate the upper bound for all the arms
        theta_t = self.zeros(self.K + 1)
        theta_t[0] = self.mu0
        theta_t[1:] = self.mean_reward_[1:] + Delta_t

        # the arms with highest upper confidence bound
        J_t = np.argmax(theta_t) - 1

        # calculate the lower bound of the cumulative regret
        xi_t = self.pulling_times_ * lambda_t + lambda_t(J_t + 1) - (1 - self.alpha) * self.t * self.mu0
        if xi_t >= 0:
            action = J_t
        else:
            action = 0

        self.action_.append(action)
        return action

    def observe(self, reward):
        self.t += 1
        pulling = self.pulling_times_[self.action_[-1] + 1]
        mean_r = self.mean_reward_[self.action_[-1] + 1]
        self.mean_reward_[self.action_[-1] + 1] = pulling / (pulling + 1) * mean_r + reward / (pulling + 1)
        self.pulling_times_[self.action_[-1] + 1] += 1
        self.total_reward_[self.action_[-1] + 1] += reward

    def rad_1(self, s: Union[np.int32, np.int64, int, np.ndarray]):
        """calculate the half of the length of the confidence interval

        Args:
            s (Union[np.float64, float, np.ndarray]): s is the pulling times. It can be an interger or an array
        """
        psi_s = 2 * np.log(self.K * (s**3) / self.delta)
        return psi_s

    def rad_2(self, s: Union[np.int32, np.int64, int, np.ndarray]):
        """calculate the half of the length of the confidence interval

        Args:
            s (Union[np.float64, float, np.ndarray]): s is the pulling times. It can be an interger or an array
        """
        zeta = self.K / self.delta
        psi_s = (
            np.log(np.maximum(3, np.log(zeta)))
            + np.log(2 * (np.e**2) * zeta)
            + zeta * (1 + np.log(zeta)) / (zeta - 1) / np.log(zeta) * np.maximum(np.log(np.log(1 + s)), 1)
        )
        return psi_s
