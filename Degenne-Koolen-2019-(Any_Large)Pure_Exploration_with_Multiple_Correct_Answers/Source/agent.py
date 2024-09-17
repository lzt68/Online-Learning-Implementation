import numpy as np


class Sticky_TaT(object):
    # Sticky Track-and-Stop
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        self.delta = delta
        self.K = K
        self.xi = xi

        self.mean_reward_ = np.zeros(K)
        self.sum_pulling_fraction = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        # self.survive_arms = np.arange(1, K + 1)
        self.pulling_list = [kk for kk in range(1, K + 1)]

        C = 10  # I am not sure C=10 is enough to fulfill the requirement
        self.beta = lambda x: np.log(C) + 2 * np.log(x) + np.log(1 / delta)

        self.stop = False

    def get_projection(self, w, epsilon):
        # project the w into the $[\epsilon, 1]^K \cap \Sigma_K$, through solving linear optimization problem
        # Please check README.md to see why the following codes can find the projection
        projected_w = np.zeros(self.K)
        threshold_index = w < epsilon
        projected_w[threshold_index] = epsilon

        gap = np.sum(np.maximum(epsilon - w, 0))
        projected_w[~threshold_index] = w[~threshold_index] - gap / (np.sum(~threshold_index))

        return projected_w

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) >= 1, "the algorithm stops"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def action_temp(self):
        assert not self.if_stop, "The algorithm stopped"

        # get the optimal pulling fraction based on empirical mean rewards
        # w_original = Get_w_star(self.mean_reward_)

        # project the w_original into the $[\epsilon, 1]^K \cap \Sigma_K$
        epsilon = 1 / np.sqrt(self.K**2 + self.t)
        projected_w = self.get_projection(w_original, epsilon)
        self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w

        action = np.argmax(self.sum_pulling_fraction - self.pulling_times) + 1
        self.action_.append(action)

        return action

    def observe(self, reward):
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        # calculate the arm to be pulled in the next round

        # determine whether to stop

    def if_stop(self):
        return self.stop
