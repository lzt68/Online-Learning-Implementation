import sys

sys.path.append("./Source/")

import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd
from copy import deepcopy
from typing import Callable
from typing import Union

import heapq


class ActionElimination_agent(object):
    def __init__(self, K: int, delta: float = 0.1, epsilon: float = 0.01) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        self.K = K
        self.delta = delta
        self.epsilon = epsilon

        # history and status
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
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        if len(self.pulling_list) == 0:
            # determine whether to conduct elimination
            # C = 2*np.array([self.U(t=self.pulling_times_[arm-1], delta=self.delta/self.K) for arm in self.survive_arms])
            C = self.U(t=self.pulling_times_[self.survive_arms - 1], delta=self.delta / self.K)
            upper_bound = self.mean_reward_[self.survive_arms - 1] + C
            lower_bound = self.mean_reward_[self.survive_arms - 1] - C

            reference_arm = np.argmax(upper_bound)
            self.survive_arms = np.array([self.survive_arms[kk] for kk in range(len(self.survive_arms)) if upper_bound[kk] > lower_bound[reference_arm]])

            self.pulling_list = list(self.survive_arms)

    def U(self, t, delta):
        e = self.epsilon
        # U_t_delta = (1 + np.sqrt(e)) * np.sqrt((1 + e) * t * np.log(np.log((1 + e) * t + 2) / delta) / 2 / t)
        U_t_delta = (1 + np.sqrt(e)) * np.sqrt((1 + e) * np.log(np.log((1 + e) * t + 2) / delta) / 2 / t)
        return U_t_delta

    def if_stop(self):
        return len(self.survive_arms) == 1

    def predict(self):
        assert len(self.survive_arms) == 1, "the algorithm doesn't stop"
        return self.survive_arms[0]


class UCB_agent(object):
    def __init__(self, K: int, delta: float = 0.1, epsilon: float = 0.01, beta: float = 1.66) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        self.K = K
        self.delta = delta
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = ((2 + beta) / beta) ** 2 * (1 + np.log(2 * np.log(((2 + beta) / beta) ** 2 * K / delta)) / np.log(K / delta))

        # history and status
        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.pulling_list = [1]

        # only when one arm get pulled, its upper bound would change
        # thus we can build a maximum heap to save the time when calculating the next pulling arm
        self.bound_ = []  # B_{i,0} = +\infty
        for kk in range(2, K + 1):
            ## heappush always return the minimum point
            ## thus we take opposite number
            heapq.heappush(self.bound_, (-9999.0, kk))
        self.max_pulling_times = 0

    def action(self):
        assert len(self.pulling_list) > 0, "pulling list is empty"
        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        if self.pulling_times_[arm - 1] > self.max_pulling_times:
            self.max_pulling_times = self.pulling_times_[arm - 1]
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        # determine the next pulling arm
        assert len(self.pulling_list) == 0, "Some arms don't get pulled"
        C = self.U(t=self.pulling_times_[arm - 1], delta=self.delta / self.K) * (1 + self.beta)
        new_upper_bound = C + self.mean_reward_[arm - 1]
        nextarm = heapq.heappushpop(self.bound_, (-new_upper_bound, arm))
        self.pulling_list.append(nextarm[1])

    def if_stop(self):
        if self.t <= self.K + 1:
            return False
        return self.max_pulling_times > self.alpha * (self.t - 1 - self.max_pulling_times)

    def U(self, t, delta):
        e = self.epsilon
        # U_t_delta = (1 + np.sqrt(e)) * np.sqrt((1 + e) * t * np.log(np.log((1 + e) * t + 2) / delta) / 2 / t)
        U_t_delta = (1 + np.sqrt(e)) * np.sqrt((1 + e) * np.log(np.log((1 + e) * t + 2) / delta) / 2 / t)
        return U_t_delta

    def predict(self):
        assert self.if_stop(), "The algorithm doesn't stop"
        max_pulling_times_index = np.argmax(self.pulling_times_) + 1
        return max_pulling_times_index


class LUCB_agent(object):
    def __init__(self, K: int, delta: float = 0.1, epsilon: float = 0.01) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        self.K = K
        self.delta = delta
        self.epsilon = epsilon

        # history and status
        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.pulling_list = [kk for kk in range(1, K + 1)]

    def action(self):
        assert len(self.pulling_list) > 0, "pulling list is empty"
        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        if len(self.pulling_list) == 0:
            # $h_t=\arg\max_{i\in[n]}\hat{\mu}_{i,T_i(t)}$
            # $l_t=\arg\max_{i\in[n]\ \{h_t\}}\hat{\mu}_{i,T_i(t)}+U(T_i(t),\delta/K)$
            assert np.all(self.pulling_times_ >= 1), "Some arms never get pulled"
            ht = np.argmax(self.mean_reward_) + 1

            C = self.U(t=self.pulling_times_, delta=self.delta / self.K)
            upper_bound = self.mean_reward_ + C
            upper_bound[ht - 1] = upper_bound[ht - 2] - 1  # to make sure ht is not lt
            lt = np.argmax(upper_bound) + 1
            assert ht != lt, "Generate same arms"

            self.pulling_list.append(ht)
            self.pulling_list.append(lt)

    def if_stop(self):
        if self.t <= self.K + 1:
            return False

        assert np.all(self.pulling_times_ >= 1), "Some arms never get pulled"
        ht = np.argmax(self.mean_reward_) + 1

        C = self.U(t=self.pulling_times_, delta=self.delta / self.K)
        upper_bound = self.mean_reward_ + C
        lower_bound = self.mean_reward_ - C

        upper_bound_musk = deepcopy(upper_bound)
        upper_bound_musk[ht - 1] = upper_bound_musk[ht - 2] - 1  # to make sure ht is not lt
        lt = np.argmax(upper_bound_musk) + 1
        assert ht != lt, "Generate same arms"

        return lower_bound[ht - 1] > upper_bound[lt - 1]

    def U(self, t, delta):
        e = self.epsilon
        # U_t_delta = (1 + np.sqrt(e)) * np.sqrt((1 + e) * t * np.log(np.log((1 + e) * t + 2) / delta) / 2 / t)
        U_t_delta = (1 + np.sqrt(e)) * np.sqrt((1 + e) * np.log(np.log((1 + e) * t + 2) / delta) / 2 / t)
        return U_t_delta

    def predict(self):
        assert self.if_stop(), "The algorithm doesn't stop"
        max_mean_reward_index = np.argmax(self.mean_reward_) + 1
        return max_mean_reward_index


#%% unit test 1, debug ActionElimination_agent
# from env import Environment_Gaussian

# K = 6
# reward = np.linspace(1.0, 0.0, K)
# random_seed = 10

# env = Environment_Gaussian(rlist=reward, K=K, random_seed=random_seed)
# agent = ActionElimination_agent(K=K)
# maximal_running = 10000
# count = 0
# while not agent.if_stop():
#     arm = agent.action()
#     reward = env.response(arm)
#     agent.observe(reward)
#     count += 1
#     if count > maximal_running:
#         break
# prediction = agent.predict()
# print(f"Predicted best arm is {prediction}, round number is {agent.t}")

#%% unit test 2, debug UCB_agent
# from env import Environment_Gaussian

# K = 10
# reward = np.linspace(1.0, 0.0, K)

# env = Environment_Gaussian(rlist=reward, K=K)
# agent = UCB_agent(K=K)
# maximal_running = 10000
# count = 0
# while not agent.if_stop():
#     arm = agent.action()
#     reward = env.response(arm)
#     agent.observe(reward)
#     count += 1
#     if count > maximal_running:
#         break
# prediction = agent.predict()
# print(f"Predicted best arm is {prediction}, round number is {agent.t}")

#%% unit test 3, debug LUCB_agent
# from env import Environment_Gaussian

# K = 11
# reward = np.linspace(1.0, 0.0, K)

# env = Environment_Gaussian(rlist=reward, K=K)
# agent = LUCB_agent(K=K)
# maximal_running = 10000
# count = 0
# while not agent.if_stop():
#     arm = agent.action()
#     reward = env.response(arm)
#     agent.observe(reward)
#     count += 1
#     if count > maximal_running:
#         break
# prediction = agent.predict()
# print(f"Predicted best arm is {prediction}, round number is {agent.t}")
