import numpy as np


class Sticky_TaT(object):
    # Sticky Track-and-Stop
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        self.delta = delta
        self.K = K
        self.xi = xi

        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.survive_arms = np.arange(1, K + 1)
        self.pulling_list = [kk for kk in range(1, K + 1)]

        C = 10  # I am not sure C=10 is enough to fulfill the requirement
        self.beta = lambda x: np.log(C) + 2 * np.log(x) + np.log(1 / delta)

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        pass
        # arm = self.pulling_list.pop(0)
        # self.action_.append(arm)
        # return arm

    def observe(self, reward):
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1
        pass

    def if_stop(self):
        return self.stop
