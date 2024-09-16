import numpy as np


class HDoC_Kano(object):
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

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1])
        prob_upper = self.mean_reward_[arm - 1] + prob_conf_len
        prob_lower = self.mean_reward_[arm - 1] - prob_conf_len
        if prob_upper < self.xi:
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm
        if prob_lower >= self.xi:
            output_arm = arm
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm

        if len(self.pulling_list) == 0:
            # determine the next arm to pull
            conf_len = self.U(Nt=self.pulling_times_[self.survive_arms - 1])
            upper_bound = self.mean_reward_[self.survive_arms - 1] + conf_len
            hat_a = self.survive_arms[np.argmax(upper_bound)]
            self.pulling_list.append(hat_a)

        return output_arm

    def U(self, Nt):
        U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        return U_t

    def U_highprob(self, Nt):
        U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


class LUCB_G_Kano(object):
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

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1])
        prob_upper = self.mean_reward_[arm - 1] + prob_conf_len
        prob_lower = self.mean_reward_[arm - 1] - prob_conf_len
        if prob_upper < self.xi:
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm
        if prob_lower >= self.xi:
            output_arm = arm
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm

        if len(self.pulling_list) == 0:
            # determine the arm to pull
            conf_len = self.U_highprob(Nt=self.pulling_times_[self.survive_arms - 1])
            prob_upper = self.mean_reward_[self.survive_arms - 1] + conf_len
            hat_a = self.survive_arms[np.argmax(prob_upper)]
            self.pulling_list.append(hat_a)

        return output_arm

    def U(self, Nt):
        U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        return U_t

    def U_highprob(self, Nt):
        U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


class APT_G_Kano(object):
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

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1])
        prob_upper = self.mean_reward_[arm - 1] + prob_conf_len
        prob_lower = self.mean_reward_[arm - 1] - prob_conf_len
        if prob_upper < self.xi:
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm
        if prob_lower >= self.xi:
            output_arm = arm
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm

        if len(self.pulling_list) == 0:
            # determine the next arm to pull
            beta_t = np.sqrt(self.pulling_times_[self.survive_arms - 1]) * np.abs(
                self.mean_reward_[self.survive_arms - 1] - self.xi
            )
            hat_a = self.survive_arms[np.argmin(beta_t)]

            self.pulling_list.append(hat_a)

        return output_arm

    def U(self, Nt):
        U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        return U_t

    def U_highprob(self, Nt):
        U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


# %% unit test 1, test HDoC_Kano
# from env import Environment_Bernoulli

# rlist = [0.1, 0.2, 0.7, 0.8]
# K = len(rlist)
# xi = 0.5
# delta = 0.1

# env = Environment_Bernoulli(rlist=rlist, K=K, random_seed=12345)
# agent = HDoC_Kano(K=K, delta=delta, xi=xi)
# output_list = []
# output_time = []
# while not agent.stop:
#     arm = agent.action()
#     reward = env.response(arm)
#     output_arm = agent.observe(reward)
#     if output_arm is not None:
#         output_list.append(output_arm)
#         output_time.append(agent.t)
# output_time.append(agent.t)
# print(f"output time is {output_time}, output arms are {output_list}")

# %% unit test 2, test LUCB_G_Kano
# from env import Environment_Bernoulli

# rlist = [0.1, 0.2, 0.7, 0.8]
# K = len(rlist)
# xi = 0.5
# delta = 0.1

# env = Environment_Bernoulli(rlist=rlist, K=K, random_seed=12345)
# agent = LUCB_G_Kano(K=K, delta=delta, xi=xi)
# output_list = []
# output_time = []
# while not agent.stop:
#     arm = agent.action()
#     reward = env.response(arm)
#     output_arm = agent.observe(reward)
#     if output_arm is not None:
#         output_list.append(output_arm)
#         output_time.append(agent.t)
# output_time.append(agent.t)
# print(f"output time is {output_time}, output arms are {output_list}")

# %% unit test 3, test APT_G_Kano
# from env import Environment_Bernoulli

# rlist = [0.1, 0.2, 0.7, 0.8]
# K = len(rlist)
# xi = 0.5
# delta = 0.1

# env = Environment_Bernoulli(rlist=rlist, K=K, random_seed=12345)
# agent = APT_G_Kano(K=K, delta=delta, xi=xi)
# output_list = []
# output_time = []
# while not agent.stop:
#     arm = agent.action()
#     reward = env.response(arm)
#     output_arm = agent.observe(reward)
#     if output_arm is not None:
#         output_list.append(output_arm)
#         output_time.append(agent.t)
# output_time.append(agent.t)
# print(f"output time is {output_time}, output arms are {output_list}")

# %% unit test 4, experiment script
# from env import Environment_Bernoulli
# from tqdm import tqdm

# # use Threshold 1 setting
# K = 10
# rlist = np.ones(10)
# rlist[0:3] = 0.1
# rlist[3:7] = 0.35 + 0.1 * np.arange(4)
# rlist[7:10] = 0.9
# xi = 0.5
# delta = 0.05

# n_exp = 1000
# output_time_ = np.zeros(
#     (n_exp, K)
# )  # if correct, there should be only 5 output (not include stop)
# stop_time_ = np.zeros(n_exp)
# correctness_ = np.ones(n_exp)
# for exp_id in tqdm(range(n_exp)):
#     env = Environment_Bernoulli(rlist=rlist, K=K, random_seed=exp_id)
#     agent = HDoC_Kano(K=K, delta=delta, xi=xi)
#     count_stop = 0
#     output_list = []
#     while not agent.stop:
#         arm = agent.action()
#         reward = env.response(arm)
#         output_arm = agent.observe(reward)
#         if output_arm is not None:
#             output_list.append(output_arm)
#             output_time_[exp_id, count_stop] = agent.t
#             count_stop += 1
#     stop_time_[exp_id] = agent.t
#     if np.any(np.sort(output_list) != np.arange(6, 11)):
#         correctness_[exp_id] = 0
# mean_output_time = np.mean(output_time_, axis=0)
# mean_stop_time = np.mean(stop_time_)
# mean_success = np.mean(correctness_)
# print(f"output time is {mean_output_time}, mean stop time is {mean_stop_time}")
# print(f"mean correctness rate is {mean_success}")
