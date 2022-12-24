# source file of agent

import numpy as np
import heapq


class Uniform_Agent:  # use round robin to pull each arm
    def __init__(self, K=2, C=10) -> None:
        """Construct an instance of uniformly pulling policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
        """
        self.K = K
        self.C = C
        self.t = 0  # index of epoch
        self.arm_ = list()  # record the action in each epoch
        # self.demand_ = dict()  # record the consumption in each epoch
        # self.reward_ = dict()  # record the observed reward in each epoch
        # for arm_index in range(1, K + 1):
        #     # for each arm, create a list
        #     self.demand_[arm_index] = list()
        #     self.reward_[arm_index] = list()
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.mean_reward_ = np.ones(K) * (-99)

    def action(self):
        # return the pulling arm in this epoch
        arm = self.t % self.K + 1
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # recorded the observed demand and reward
        # self.reward_[self.arm_[-1]].append(reward)
        # self.demand_[self.arm_[-1]].append(demand)
        self.t = self.t + 1
        self.pulling_times_[self.arm_[-1] - 1] += 1
        self.total_reward_[self.arm_[-1] - 1] += reward
        self.mean_reward_[self.arm_[-1] - 1] = self.total_reward_[self.arm_[-1] - 1] / self.pulling_times_[self.arm_[-1] - 1]

    def predict(self):
        # output the predicted best arm, we need to make sure the pulling times of each arm is the same
        # pulling_times = self.t // self.K
        # mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in range(1, self.K + 1)])
        best_arm = np.argmax(self.mean_reward_) + 1
        return best_arm


class UCB_Agent:
    def __init__(self, K=2, C=10, a=2) -> None:
        """Construct an instance of Upper Confidence Bound Policy,
        see Bubeck et.al 2009, Pure Exploration in Multi-armed Bandits Problems
        DOI: 10.1007/978-3-642-04414-4_7

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
            a (int, optional): The exploration factor. Defaults to 2.
        """
        self.K = K
        self.C = C
        self.a = a
        self.t = 1
        self.pulling_times_ = np.zeros(K)
        self.arm_ = list()  # record the action in each epoch
        self.total_consumption = 0

        self.demand_ = np.zeros(K)  # total consumption
        self.reward_ = np.zeros(K)  # total reward
        self.mean_reward_ = np.zeros(K)
        self.confidence_ = np.ones(K) * 9999

        self.J = 1  # recommended best arm
        self.J_ = list()

    def action(self):
        # return the pulling arm in this epoch
        upper_bound_ = self.mean_reward_ + self.confidence_
        arm = np.argmax(upper_bound_) + 1
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this round
        arm_index = self.arm_[-1] - 1

        # update the history of this arm
        self.reward_[arm_index] += reward
        self.demand_[arm_index] += demand
        self.total_consumption += demand
        self.pulling_times_[arm_index] += 1
        self.mean_reward_[arm_index] = self.reward_[arm_index] / self.pulling_times_[arm_index]
        self.t += 1

        # generate a new arm
        if self.t <= self.K + 1:
            # some of the pulling times equal to zero
            index = self.pulling_times_ > 0
            self.confidence_[index] = np.sqrt(self.a * np.log(self.t) / self.pulling_times_[index])
        else:
            self.confidence_ = np.sqrt(self.a * np.log(self.t) / self.pulling_times_)

        # calculate the predicted arm
        self.J = np.argmax(self.mean_reward_) + 1
        self.J_.append(self.J)

    def predict(self):
        # output the predicted best arm
        arm = self.J
        return arm


class SequentialHalving_FixedBudget_Agent:
    def __init__(self, K=2, budget=10) -> None:
        """The classic Fixed Budget SH algorithm, in this case, the consumption is always 1

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            Budget (int, optional): Available budget Defaults to 10.
        """
        assert np.ceil(np.log2(K)) * K <= budget, "budget is not enough"

        self.K = K
        self.budget = budget
        self.t = 0  # index of round
        self.t_q = 0  # index of round in each phase
        self.q = 0  # index of phase
        self.arm_ = list()  # record the action in each epoch

        self.reward_ = dict()  # record the observed reward of arms in each epoch
        # self.mean_reward_ = np.zeros(K)  # record the mean reward
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # when we enter a new epoch, we clear the existing memory
            self.reward_[arm_index] = list()
        self.pulling_times_ = np.zeros(K)

        self.total_reward_ = dict()
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # but we will not clear the memory
            self.total_reward_[arm_index] = list()

        self.survive_arms = list(range(1, K + 1))

        self.pulling_list = []
        for kk in range(1, K + 1):
            self.pulling_list = self.pulling_list + [kk] * int(np.floor(self.budget / np.ceil(np.log2(self.K)) / self.K))

        self.complete = False  # mark whether the algorithm complete or not

    def action(self):
        # return the pulling arm in this epoch
        assert len(self.pulling_list) > 0
        arm = self.pulling_list[0]
        self.pulling_list.pop(0)
        self.arm_.append(arm)
        return arm

    def observe(self, reward):
        # record the arms in this phase
        self.reward_[self.arm_[-1]].append(reward)
        self.pulling_times_[self.arm_[-1] - 1] += 1

        # record the arms in this overall array
        self.total_reward_[self.arm_[-1]].append(reward)

        # update the index of rounds
        self.t = self.t + 1
        self.t_q = self.t_q + 1

        if len(self.survive_arms) == 1:
            self.complete = True
            return

        # check whether conduct the elimination
        if len(self.pulling_list) == 0:
            # we need to make sure all the arm share the same pulling times
            pulling_times = self.t_q // len(self.survive_arms)
            mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in self.survive_arms])
            # sort the mean reward with descending order
            sort_order = np.argsort(mean_reward)[::-1]
            self.survive_arms = np.array([self.survive_arms[ii] for ii in sort_order[: int(np.ceil(len(self.survive_arms) / 2))]])
            self.survive_arms = np.sort(self.survive_arms)

            # generate pulling list
            self.pulling_list = []
            for arm in self.survive_arms:
                self.pulling_list = self.pulling_list + [arm] * int(np.floor(self.budget / np.ceil(np.log2(self.K)) / len(self.survive_arms)))

            # clear the memory
            self.t_q = 0
            for arm_index in range(1, self.K + 1):
                self.reward_[arm_index] = list()
            self.consumption = 0
            self.q = self.q + 1

    def predict(self):
        # output the predicted best arm
        assert len(self.survive_arms) == 1
        best_arm = self.survive_arms[0]
        return best_arm

    def if_complete(self):
        return self.complete


class DoublingSequentialHalving_ATMean_Agent:
    # Apply doubling trick on Fixed budget sequential halving, to make it an anytime algorithm
    def __init__(self, K=2, C=10) -> None:
        """Construct an instance of Sequential Halving policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
        """
        self.K = K
        self.C = C
        self.t = 0  # index of round
        self.arm_ = list()  # record the action in each epoch

        self.demand_ = dict()  # record the consumption of arms in each epoch
        self.reward_ = dict()  # record the observed reward of arms in each epoch
        self.consumption = 0  # record the consumption in each phase
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            self.demand_[arm_index] = list()
            self.reward_[arm_index] = list()

        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)  # the cumulative reward of each arm
        self.mean_reward = np.zeros(K)
        self.total_consumption = 0  # the total consumption of all the phase

        # setup the SH oracle
        self.budget = K * np.ceil(np.log2(K))
        self.SH_oracle = SequentialHalving_FixedBudget_Agent(K=K, budget=self.budget)

    def action(self):
        # return the pulling arm in this epoch
        arm = self.SH_oracle.action()
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this round
        self.reward_[self.arm_[-1]].append(reward)
        self.demand_[self.arm_[-1]].append(demand)

        # update the empirical mean reward of this arm
        self.pulling_times_[self.arm_[-1] - 1] += 1
        self.total_reward_[self.arm_[-1] - 1] += reward
        self.mean_reward[self.arm_[-1] - 1] = self.total_reward_[self.arm_[-1] - 1] / self.pulling_times_[self.arm_[-1] - 1]

        # update the consumption
        self.total_consumption = self.total_consumption + demand

        # update the index of rounds
        self.t = self.t + 1

        # update the history of SH oracle
        self.SH_oracle.observe(reward=reward)
        if self.SH_oracle.if_complete():
            self.budget *= 2
            self.SH_oracle = SequentialHalving_FixedBudget_Agent(K=self.K, budget=self.budget)

    def predict(self):
        # output the predicted best arm
        best_arm = np.argmax(self.mean_reward) + 1
        return best_arm


class AT_LUCB_Agent:
    def __init__(self, K=2, C=10, delta_1=0.5, alpha=0.99, epsilon=0.0, m=1) -> None:
        """Construct an instance of Anytime Lower and Upper Confidence Bound
        The algorithm came from June&Nowak2016, top m identification problem

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
            delta_1 (float, optional): Confidence Level. Defaults to 0.5, 1/200 <= delta_1 <= n
            alpha (float, optional): Discount Factor. Defaults to 0.99., 1/50 <= alpha < 1
            epsilon (float, optional): Tolerance of error. Defaults to 0.0.
            m (int, optional): we aim to find top m arms. Default value is 1
        """
        self.K = K
        self.C = C
        self.delta_1 = delta_1
        self.alpha = alpha
        self.epsilon = epsilon
        self.m = m

        self.t = 0  # index of round
        self.t_for_delta = 1  # in the algorithm, t_for_delta increases only after 2 pulls
        self.total_consumption = 0  # record the overall consumption in each phase
        self.S = 1  # S(0)
        self.S_ = list()  # record the chaning history of S
        self.J = np.arange(1, self.m + 1)
        self.J_ = list()  # record the chaning history of J

        self.bound_ = np.ones(K) * 9999  # B_{i,0} = +\infty
        self.pulling_times_ = np.zeros(K)
        self.demand_ = np.zeros(K)  # total consumption
        self.reward_ = np.zeros(K)  # total reward
        self.mean_reward_ = np.zeros(K)
        self.arm_ = list()  # record the action in each epoch

        self.pulling_list = list(np.arange(1, K + 1))
        # Each time, this algorithm will generate two arms to pull

    def action(self):
        # return the pulling arm in this epoch
        assert len(self.pulling_list) > 0, "failed to generate pulling arms"
        arm = self.pulling_list[0]
        self.pulling_list.pop(0)
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this round
        arm_index = self.arm_[-1] - 1

        # record the arms in this round
        self.reward_[arm_index] += reward
        self.demand_[arm_index] += demand
        self.total_consumption += demand
        self.pulling_times_[arm_index] += 1
        self.mean_reward_[arm_index] = self.reward_[arm_index] / self.pulling_times_[arm_index]

        # update the index of rounds
        self.t = self.t + 1
        if self.t <= self.K - 1:
            # there are still some arms that had never been pulled
            return

        # if self.pulling_list is empty, we need to regenerate arms
        if len(self.pulling_list) == 0:
            delta_s_t_1 = self.delta_1 * self.alpha ** (self.S - 1)
            if self.Term(self.t_for_delta, delta_s_t_1, self.epsilon):
                ## update S(t) and new pulling arms
                self.S, ht_star_delta, lt_star_delta, self.J = self.UpdateS(self.S)
                self.pulling_list.append(ht_star_delta)
                self.pulling_list.append(lt_star_delta)
                self.t_for_delta += 1
            else:
                if self.S == 1:
                    self.J = np.argpartition(self.mean_reward_, -self.m)[-self.m :] + 1

                # generate new pulling arms
                _, _, ht_star_delta, lt_star_delta = self.Get_LUCB_l_h_star(self.t_for_delta, self.delta_1 * self.alpha ** (self.S - 1))
                self.pulling_list.append(ht_star_delta)
                self.pulling_list.append(lt_star_delta)
                self.t_for_delta += 1

            self.S_.append(self.S)  # record the history
            self.J_.append(self.J)

    def predict(self, m=1):
        # output the predicted best arm
        if m == 1:
            arm = np.argmax(self.mean_reward_) + 1
        else:
            arm = self.J
        return arm

    def UpdateS(self, S):
        # This fcuntion is used to accelerate the step to update S

        # calculate set High^t
        hight = np.argpartition(self.mean_reward_, -self.m)[-self.m :]
        mask_hight = np.ones(self.K, dtype=bool)
        mask_hight[hight] = False
        mask_nothight = ~mask_hight

        tempS = S + 1
        while True:
            delta = self.delta_1 * self.alpha ** (tempS - 1)
            confidence = np.sqrt(1 / self.pulling_times_ / 2 * (np.log(5 * self.K / 4 / delta) + 4 * np.log(self.t_for_delta)))

            # upper and lower confidence bound
            ut_a_delta = self.mean_reward_ + confidence
            lt_a_delta = self.mean_reward_ - confidence

            # get h^t_{*}(\delta)
            lt_a_delta_mask = np.ma.array(lt_a_delta, mask=mask_hight)
            ht_star_delta = np.argmin(lt_a_delta_mask) + 1

            # get l^t_{*}(\delta)
            ut_a_delta_mask = np.ma.array(ut_a_delta, mask=mask_nothight)
            lt_star_delta = np.argmax(ut_a_delta_mask) + 1

            if ut_a_delta[lt_star_delta - 1] - lt_a_delta[ht_star_delta - 1] >= self.epsilon:
                break

            tempS += 1

        return tempS, ht_star_delta, lt_star_delta, hight + 1

    def Get_LUCB_l_h_star(self, t, delta):
        # auxiliary function, help to calculate $U^t_a(\delta), L^t_a(\delta), h^t_*(\delta), l^t_*(\delta)$

        ## calculate the upper and lower confidence bound
        confidence = np.array([np.sqrt(1 / self.pulling_times_[kk - 1] / 2 * np.log(5 * self.K * t**4 / 4 / delta)) if self.pulling_times_[kk - 1] > 0 else 9999 for kk in range(1, self.K + 1)])
        ut_a_delta = self.mean_reward_ + confidence
        lt_a_delta = self.mean_reward_ - confidence

        ## calculate the $h^t_*(\delta)$
        hight = np.argpartition(self.mean_reward_, -self.m)[-self.m :]  # index of top m greatest value
        mask = np.ones(self.K, dtype=bool)
        mask[hight] = False
        lt_a_delta_mask = np.ma.array(lt_a_delta, mask=mask)
        ht_star_delta = np.argmin(lt_a_delta_mask) + 1

        ## calculate the $l^t_*(\delta)$
        mask = ~mask
        ut_a_delta_mask = np.ma.array(ut_a_delta, mask=mask)
        lt_star_delta = np.argmax(ut_a_delta_mask) + 1

        return ut_a_delta, lt_a_delta, ht_star_delta, lt_star_delta

    def Term(self, t, delta, epsilon):
        # auxiliary function
        # use it to judge whether $U^t_{l^t_*(\delta)}(\delta)-L^t_{h^t_*(\delta)}(\delta)<epsilon$
        ut_a_delta, lt_a_delta, ht_star_delta, lt_star_delta = self.Get_LUCB_l_h_star(t, delta)
        return ut_a_delta[lt_star_delta - 1] - lt_a_delta[ht_star_delta - 1] < epsilon


#%% unit test 1
# from env import Env_FixedConsumption

# K = 10
# # C = np.ceil(np.log2(K)) * K
# C = 1000
# np.random.seed(7)
# p_list = np.random.uniform(low=0.0, high=1.0, size=K)
# d_list = np.ones(K)
# random_seed = 0
# env = Env_FixedConsumption(p_list, d_list, K, C, random_seed)
# agent = SequentialHalving_FixedBudget_Agent(K, C)
# while not env.if_stop():
#     arm = agent.action()
#     consumption, reward = env.response(arm)
#     agent.observe(reward)
#     if agent.if_complete():
#         break
# print(f"predicted best arm is {agent.predict()}, actual best arm is {np.argmax(p_list)+1}")
# for arm_index in range(1, K + 1):
#     print(f"arm {arm_index} empirical reward is {np.mean(agent.total_reward_[arm_index])},")


#%% unit test 2
# from env import Env_FixedConsumption

# K = 10
# # C = np.ceil(np.log2(K)) * K
# C = 1000
# np.random.seed(10)
# p_list = np.random.uniform(low=0.0, high=1.0, size=K)
# d_list = np.ones(K)
# random_seed = 0
# env = Env_FixedConsumption(p_list, d_list, K, C, random_seed)
# agent = UCB_Agent(K, C)
# while not env.if_stop():
#     arm = agent.action()
#     consumption, reward = env.response(arm)
#     agent.observe(consumption, reward)

# print(f"predicted best arm is {agent.predict()}, actual best arm is {np.argmax(p_list)+1}")
# # for arm_index in range(1, K + 1):
# #     print(f"arm {arm_index} empirical reward is {np.mean(agent.mean_reward_[arm_index-1])}, actual is {p_list[arm_index-1]}")


#%% unit test 3
# from env import Env_FixedConsumption

# K = 10
# # C = np.ceil(np.log2(K)) * K
# C = 1000
# np.random.seed(4)
# p_list = np.random.uniform(low=0.0, high=1.0, size=K)
# d_list = np.ones(K)
# random_seed = 0
# env = Env_FixedConsumption(p_list, d_list, K, C, random_seed)
# agent = DoublingSequentialHalving_ATMean_Agent(K, C)
# while not env.if_stop():
#     arm = agent.action()
#     consumption, reward = env.response(arm)
#     agent.observe(consumption, reward)
# print(f"predicted best arm is {agent.predict()}, actual best arm is {np.argmax(p_list)+1}")
# # for arm_index in range(1, K + 1):
# #     print(f"arm {arm_index} empirical reward is {np.mean(agent.mean_reward[arm_index-1])}, actual is {p_list[arm_index-1]}")
