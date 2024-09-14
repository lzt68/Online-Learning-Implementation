import numpy as np


class HDoC(object):
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"

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

    def action(self):
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) > 1, "the algorithm stops"
        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert len(self.survive_arms) > 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = (
            self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        )
        self.t += 1

        if len(self.pulling_list) == 0:
            # determine the arm to pull
            conf_len = self.U(t=self.pulling_times_[self.survive_arms - 1])
            upper_bound = self.mean_reward_[self.survive_arms - 1] + conf_len

            prob_conf_len = self.U(t=self.pulling_times_[self.survive_arms - 1])
            prob_upper = self.mean_reward_[self.survive_arms - 1] + prob_conf_len
            prob_lower = self.mean_reward_[self.survive_arms - 1] - prob_conf_len

            hat_a = np.argmax(upper_bound)
            self.pulling_list.append(hat_a)

            # determine whether to output arm or eliminate arm

    def U(self, Nt):
        U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        return U_t

    def U_highprob(self, Nt):
        U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)
        return U_t_delta


print(np.array([1, 2, 3]))
# class Uniform_Agent(object):
#     def __init__
