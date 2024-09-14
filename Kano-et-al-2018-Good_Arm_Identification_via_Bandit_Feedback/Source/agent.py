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

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) >= 1, "the algorithm stops"

        # for robustness, check whether all the high porb upper bound
        # of suriving arms are below xi
        temp_if_stop = True
        for arm in self.survive_arms:
            prob_conf_len = self.U_highprob(
                Nt=np.maximum(1, self.pulling_times_[arm - 1])
            )
            if self.mean_reward_[arm - 1] + prob_conf_len >= self.xi:
                temp_if_stop = False
                break
        if temp_if_stop:
            self.stop = True
            return None

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = (
            self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        )
        self.t += 1

        if len(self.pulling_list) == 0:
            # determine the arm to pull
            conf_len = self.U(Nt=self.pulling_times_[self.survive_arms - 1])
            upper_bound = self.mean_reward_[self.survive_arms - 1] + conf_len
            hat_a = self.survive_arms[np.argmax(upper_bound)]
            self.pulling_list.append(hat_a)

            output_arm = None
            prob_conf_len = self.U_highprob(Nt=self.pulling_times_[hat_a - 1])
            prob_upper = self.mean_reward_[hat_a - 1] + prob_conf_len
            prob_lower = self.mean_reward_[hat_a - 1] - prob_conf_len
            if prob_upper < self.xi:
                self.survive_arms = self.survive_arms[self.survive_arms != hat_a]
                if len(self.survive_arms) == 0:
                    self.stop = True

            if prob_lower >= self.xi:
                output_arm = hat_a
                self.survive_arms = self.survive_arms[self.survive_arms != hat_a]
                if len(self.survive_arms) == 0:
                    self.stop = True

            return output_arm

    def U(self, Nt):
        U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        return U_t

    def U_highprob(self, Nt):
        U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


# unit test 1
from env import Environment_Bernoulli

rlist = [0.1, 0.2, 0.7, 0.8]
K = len(rlist)
xi = 0.5
delta = 0.1

env = Environment_Bernoulli(rlist=rlist, K=K, random_seed=12345)
agent = HDoC(K=K, delta=delta, xi=xi)
output_list = []
output_time = []
while not agent.stop:
    arm = agent.action()
    reward = env.response(arm)
    output_arm = agent.observe(reward)
    if output_arm is not None:
        output_list.append(output_arm)
        output_time.append(agent.t)
output_time.append(agent.t)
print(f"output time is {output_time}, output arms are {output_list}")
