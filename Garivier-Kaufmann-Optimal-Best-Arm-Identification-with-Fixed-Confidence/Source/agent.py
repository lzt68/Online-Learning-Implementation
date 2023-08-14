from typing import Union

import numpy as np
from numpy import linalg
from copy import deepcopy
from scipy.optimize import minimize, linprog

from utils import d_fun, I, g, x_fun, F_fun, Get_w_star


class D_Tracking(object):
    def __init__(self, K: int = 4, delta: Union[float, np.float64] = 0.1) -> None:
        """Adopt C-Tracking rule to pull arms

        Args:
            K (int, optional): Number of arms. Defaults to 4.
            delta (Union[float, np.float64], optional): Threshold of failure probability. Defaults to 0.1.
        """
        self.K = K
        self.delta = delta

        # generate dictionary to record realized reward and consumption
        self.pulling_times = np.zeros(K)
        self.mean_reward_ = np.zeros(K)
        self.mean_consumption_ = np.zeros(K)

        self.action_ = list()
        self.reward_ = dict()
        self.consumption_ = dict()
        for kk in range(1, K + 1):
            self.reward_[kk] = list()
            self.consumption_[kk] = list()

        # self.w = list()  # record the optimized pulling fraction in each round
        self.t = 1

        self.g = lambda x: np.maximum(np.sqrt(x) - (self.K + 1) / 2, 0)
        self.beta = lambda x: np.log(2 * x * (self.K - 1)) + np.log(1 / self.delta)
        self.if_stop = False

    def action(self):
        assert not self.if_stop, "The algorithm stopped"

        U_t = np.array([aa for aa in range(0, self.K) if self.pulling_times[aa] <= self.g(self.t)])
        if len(U_t) >= 1:
            arm_index = np.argmin(self.pulling_times[U_t])
            action = U_t[arm_index] + 1
        else:
            w_star = Get_w_star(self.mean_reward_)
            action = np.argmax(self.t * w_star - self.pulling_times) + 1
        self.action_.append(action)

        return action

    def observe(self, r, d):
        assert not self.if_stop, "The algorithm stopped"

        # record the observation
        action = self.action_[-1]
        self.reward_[action].append(r)
        self.consumption_[action].append(d)

        pull = self.pulling_times[action - 1]
        self.mean_reward_[action - 1] = self.mean_reward_[action - 1] * (pull / (pull + 1)) + r / (pull + 1)
        self.mean_consumption_[action - 1] = self.mean_consumption_[action - 1] * (pull / (pull + 1)) + d / (pull + 1)
        self.pulling_times[action - 1] += 1

        # judge whether we should stop
        # our stoppting rule is $\max_a\min_{b\ne a} Z_{a, b}(t) > \beta(t, \delta)$
        empirical_best = np.argmax(self.mean_reward_)

        def get_z_ab_t(best_mean_reward, best_pulling_times, mean_reward, pulling_times):
            hat_mu_ab = (best_mean_reward * best_pulling_times + mean_reward * pulling_times) / (
                best_pulling_times + pulling_times
            )
            z_ab_t = best_pulling_times * d_fun(best_mean_reward, hat_mu_ab) + pulling_times * d_fun(
                mean_reward, hat_mu_ab
            )
            return z_ab_t

        z_ = np.array(
            [
                get_z_ab_t(
                    self.mean_reward_[empirical_best],
                    self.pulling_times[empirical_best],
                    self.mean_reward_[aa],
                    self.pulling_times[aa],
                )
                for aa in range(self.K)
                if aa != empirical_best
            ]
        )
        if np.max(z_) > self.beta(self.t):
            self.if_stop = True
        self.t += 1

    def predict(self):
        best_arm = np.argmax(self.mean_reward_) + 1
        return best_arm

    def stop(self):
        return self.if_stop


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
        self.pulling_times = np.zeros(K)
        self.sum_pulling_fraction = np.zeros(K)
        self.mean_reward_ = np.zeros(K)
        self.mean_consumption_ = np.zeros(K)

        self.action_ = list()
        self.reward_ = dict()
        self.consumption_ = dict()
        for kk in range(1, K + 1):
            self.reward_[kk] = list()
            self.consumption_[kk] = list()

        # self.w = list()  # record the optimized pulling fraction in each round
        self.t = 1

        self.g = lambda x: np.maximum(np.sqrt(x) - (self.K + 1) / 2, 0)
        self.beta = lambda x: np.log(2 * x * (self.K - 1)) + np.log(1 / self.delta)
        self.if_stop = False

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
        assert not self.if_stop, "The algorithm stopped"

        # get the optimal pulling fraction based on empirical mean rewards
        w_original = Get_w_star(self.mean_reward_)

        # project the w_original into the $[\epsilon, 1]^K \cap \Sigma_K$
        epsilon = 1 / np.sqrt(self.K**2 + self.t)
        projected_w = self.get_projection(w_original, epsilon)
        self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w

        action = np.argmax(self.sum_pulling_fraction - self.pulling_times) + 1
        self.action_.append(action)

        return action

    def observe(self, r, d):
        assert not self.if_stop, "The algorithm stopped"

        # record the observation
        action = self.action_[-1]
        self.reward_[action].append(r)
        self.consumption_[action].append(d)

        pull = self.pulling_times[action - 1]
        self.mean_reward_[action - 1] = self.mean_reward_[action - 1] * (pull / (pull + 1)) + r / (pull + 1)
        self.mean_consumption_[action - 1] = self.mean_consumption_[action - 1] * (pull / (pull + 1)) + d / (pull + 1)
        self.pulling_times[action - 1] += 1

        # judge whether we should stop
        # our stoppting rule is $\max_a\min_{b\ne a} Z_{a, b}(t) > \beta(t, \delta)$
        empirical_best = np.argmax(self.mean_reward_)

        def get_z_ab_t(best_mean_reward, best_pulling_times, mean_reward, pulling_times):
            hat_mu_ab = (best_mean_reward * best_pulling_times + mean_reward * pulling_times) / (
                best_pulling_times + pulling_times
            )
            z_ab_t = best_pulling_times * d_fun(best_mean_reward, hat_mu_ab) + pulling_times * d_fun(
                mean_reward, hat_mu_ab
            )
            return z_ab_t

        z_ = np.array(
            [
                get_z_ab_t(
                    self.mean_reward_[empirical_best],
                    self.pulling_times[empirical_best],
                    self.mean_reward_[aa],
                    self.pulling_times[aa],
                )
                for aa in range(self.K)
                if aa != empirical_best
            ]
        )
        if np.max(z_) > self.beta(self.t):
            self.if_stop = True
        self.t += 1

    def predict(self):
        best_arm = np.argmax(self.mean_reward_) + 1
        return best_arm

    def stop(self):
        return self.if_stop


# %% unit test 1, how to solve the optimal pull fraction given mu
# import time

# T1 = time.time()
# K = 4
# mu = np.array([0.5, 0.45, 0.44, 0.43])
# # K = 2
# # mu = np.array([0.5, 0.45])


# def d(x, y):
#     return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


# def I(mu1, mu2, alpha):
#     mu_temp = alpha * mu1 + (1 - alpha) * mu2
#     return alpha * d(mu1, mu_temp) + (1 - alpha) * d(mu2, mu_temp)


# def g(a, x):
#     alpha = 1 / (1 + x)
#     return (1 + x) * I(mu[0], mu[a - 1], alpha)


# def x_fun(a, y, epsilon=0.001):
#     # a is in {1, 2, ..., K}
#     left = 0
#     right = 100
#     while g(a, right) < y:
#         left = right
#         right *= 2
#     while np.abs(right - left) > epsilon:
#         temp = (left + right) / 2
#         if g(a, temp) >= y:
#             right = temp
#         else:
#             left = temp
#     return (left + right) / 2


# def F_fun(y):
#     ratio_sum = 0
#     for aa in range(2, K + 1):
#         x_a_y = x_fun(a=aa, y=y)
#         temp_mu = (mu[0] + x_a_y * mu[aa - 1]) / (1 + x_a_y)
#         ratio_sum += d(mu[0], temp_mu) / d(mu[aa - 1], temp_mu)
#     return ratio_sum


# # a = 2
# # x = 10
# # y = g(a=a, x=x)
# # print(y)
# # print(x_fun(a=a, y=y))


# epsilon = 0.01
# left = 0
# right = d(mu[0], mu[1]) / 2
# while F_fun(right) < 1:
#     left = right
#     right = (d(mu[0], mu[1]) + right) / 2
# temp = (left + right) / 2
# while (np.abs(right - left) > epsilon) or (np.abs(F_fun(temp) - 1) > epsilon):
#     if F_fun(temp) >= 1:
#         right = temp
#     else:
#         left = temp
#     temp = (left + right) / 2
# y_star = (left + right) / 2

# x_a_y_star = np.array([1.0] + [x_fun(a=aa, y=y_star) for aa in range(2, K + 1)])
# w_star = x_a_y_star / np.sum(x_a_y_star)

# T2 = time.time()
# print(w_star)
# print(f"Time Consumption is {T2-T1}")

# %% unit test 2, test whether D-Tracking can work
# from env import Env__Deterministic_Consumption

# K = 2
# # mu = np.array([0.5, 0.3])
# mu = np.array([0.3, 0.5])
# delta = 0.1
# n_experiments = 10

# for exp_id in range(n_experiments):
#     count_round = 0

#     env = Env__Deterministic_Consumption(K=K, d=np.ones(K), r=mu, random_seed=exp_id)
#     agent = D_Tracking(K=K, delta=delta)

#     while not agent.if_stop:
#         action = agent.action()
#         reward, demand = env.response(action=action)
#         agent.observe(r=reward, d=demand)
#     print(f"Experiment {exp_id}, predicted best arm is {agent.predict()}")

# %% unit test 3, test whether D-Tracking can work on a larger scale
# from env import Env__Deterministic_Consumption

# K = 4
# # mu = np.array([0.5, 0.3])
# mu = np.array([0.3, 0.8, 0.4, 0.2])
# delta = 0.1
# n_experiments = 10

# for exp_id in range(n_experiments):
#     count_round = 0

#     env = Env__Deterministic_Consumption(K=K, d=np.ones(K), r=mu, random_seed=exp_id)
#     agent = D_Tracking(K=K, delta=delta)

#     while not agent.if_stop:
#         action = agent.action()
#         reward, demand = env.response(action=action)
#         agent.observe(r=reward, d=demand)
#     print(f"Experiment {exp_id}, predicted best arm is {agent.predict()}")

# %% unit test 4, test the whether we can correctly solve the linear programming problem
# K = 4
# delta = 0.1

# epsilon = 0.1
# # w = np.array([1.0, 0.0, 0.0, 0.0])
# w = np.random.uniform(size=K)
# w = w / np.sum(w)

# print(w)

# agent = C_Tracking(K=K, delta=delta)
# projected_w = agent.get_projection(w, epsilon)
# print(projected_w, np.max(np.abs(w - projected_w[0:K])))


# projected_w = np.zeros(K)
# threshold_index = w < epsilon
# projected_w[threshold_index] = epsilon
# gap = np.sum(np.maximum(epsilon - w, 0))
# projected_w[~threshold_index] = w[~threshold_index] - gap / (np.sum(~threshold_index))
# print(projected_w, np.max(np.abs(w - projected_w)))

# %% unit test 5, test whether C-Tracking can work
# from env import Env__Deterministic_Consumption

# K = 2
# # mu = np.array([0.5, 0.3])
# mu = np.array([0.3, 0.5])
# delta = 0.1
# n_experiments = 10

# for exp_id in range(n_experiments):
#     count_round = 0

#     env = Env__Deterministic_Consumption(K=K, d=np.ones(K), r=mu, random_seed=exp_id)
#     agent = C_Tracking(K=K, delta=delta)

#     while not agent.if_stop:
#         action = agent.action()
#         reward, demand = env.response(action=action)
#         agent.observe(r=reward, d=demand)
#     print(f"Experiment {exp_id}, predicted best arm is {agent.predict()}")

# %% unit test 6, test whether C-Tracking can work on a larger scale
# from env import Env__Deterministic_Consumption

# K = 4
# # mu = np.array([0.5, 0.3])
# mu = np.array([0.3, 0.8, 0.4, 0.2])
# delta = 0.1
# n_experiments = 10

# for exp_id in range(n_experiments):
#     count_round = 0

#     env = Env__Deterministic_Consumption(K=K, d=np.ones(K), r=mu, random_seed=exp_id)
#     agent = C_Tracking(K=K, delta=delta)

#     while not agent.if_stop:
#         action = agent.action()
#         reward, demand = env.response(action=action)
#         agent.observe(r=reward, d=demand)
#     print(f"Experiment {exp_id}, predicted best arm is {agent.predict()}")
# %% unit test 7, conduct the numeric experiment
# from tqdm import tqdm
# from env import Env__Deterministic_Consumption

# # conduct the experiment
# n_experiments = 10
# delta_ = [0.1, 0.05, 0.01, 0.005]

# # container of numeric record
# pulling_times = dict()
# predicted_arm = dict()
# best_arm = dict()

# pulling_times["D-Tracking"] = np.zeros((len(delta_), n_experiments))
# predicted_arm["D-Tracking"] = np.zeros((len(delta_), n_experiments))
# best_arm["D-Tracking"] = np.zeros((len(delta_), n_experiments))

# # synthesis setting
# mu = np.array([0.5, 0.2, 0.3, 0.4])
# K = 4

# # run the algorithm
# np.random.seed(12345)
# for ii, delta in enumerate(delta_):
#     for exp_id in tqdm(range(n_experiments)):
#         # shuffle the best arm
#         mu_temp = mu.copy()
#         np.random.shuffle(mu_temp)
#         best_arm["D-Tracking"][ii, exp_id] = np.argmax(mu_temp) + 1

#         # create the agent instance and run the experiment
#         env = Env__Deterministic_Consumption(K=K, d=np.ones(K), r=mu, random_seed=exp_id)
#         agent = D_Tracking(K=K, delta=delta)
#         while not agent.if_stop:
#             action = agent.action()
#             reward, demand = env.response(action=action)
#             agent.observe(r=reward, d=demand)
#         predicted_arm["D-Tracking"][ii, exp_id] = agent.predict()
#         pulling_times["D-Tracking"][ii, exp_id] = agent.t - 1
