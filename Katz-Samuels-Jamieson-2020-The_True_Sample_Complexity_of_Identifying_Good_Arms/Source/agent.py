import sys

sys.path.append("./Source/")

import numpy as np
from copy import deepcopy
from typing import Callable
from typing import Union
from numpy.random import Generator, PCG64


class BracketingUCB_epsilon_good(object):
    # the algorithm doesn't take epsilon as an input
    def __init__(self, K: int, delta: float = 0.1, random_seed=12345) -> None:
        """Algorithm 1, $\epsilon$-good arm identification

        Args:
            K (int): Number of arms.
            delta (float, optional): Confidence level. Defaults to 0.1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        # in the original paper, $n$ denotes the total number of arms
        # here we use $K$ to denote the numebr of arms

        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        self.K = K
        self.delta = delta
        # self.epsilon = epsilon
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))
        self.l = 0

        # history
        # in bracketing UCB, the algorithm will gradually creat brakets (subset of [K]) with larger size
        # and the pulling history of each each bracket are independent
        # hence we use a dictionary to contain the pulling history of each bracket
        # the key is the index of bracket
        self.mean_reward_ = dict()
        self.total_reward_ = dict()
        self.bracket_ = dict()
        self.pulling_times_ = dict()
        self.action_ = []  # each entry is $(R_t, A_t)$

        # status
        self.l = 0
        self.t = 1
        self.R = 0

    def action(self):
        if self.t >= (2**self.l) * self.l:
            M_lp1 = np.minimum(self.K, 2 ** (self.l + 1))
            bracket = np.sort(self.random_generator.choice(np.arange(1, self.K + 1), M_lp1, replace=False))
            self.bracket_[self.l + 1] = bracket

            # create dictionary for the pulling times
            self.pulling_times_[self.l + 1] = dict()
            for arm in bracket:
                self.pulling_times_[self.l + 1][arm] = 0

            # create dictionary for the mean rewards
            self.mean_reward_[self.l + 1] = dict()
            for arm in bracket:
                self.mean_reward_[self.l + 1][arm] = 0

            # create dictionary for the total rewards
            self.total_reward_[self.l + 1] = dict()
            for arm in bracket:
                self.total_reward_[self.l + 1][arm] = 0

            self.l += 1
        # self.R = ((self.R + 1) % self.l) + 1  # this is equivalent to R=1+R*(R<l)
        self.R = 1 + self.R * (self.R < self.l)  # then R will take value 1,2,3,..,l,1,2,3,...,l,... repeatedly

        # determine the pulling arm
        for ii in self.bracket_[self.R]:  # check whether there exists an arm has 0 pulling times
            if self.pulling_times_[self.R][ii] == 0:
                self.action_.append((self.R, ii))
                return ii
        arm = 0
        UCB_max = -np.infty
        for ii in self.bracket_[self.R]:  # find the arm with highest UCB
            UCB_ii = self.mean_reward_[self.R][ii] + self.U(self.pulling_times_[self.R][ii], self.delta)
            if UCB_ii > UCB_max:
                arm = ii
        self.action_.append((self.R, arm))
        return arm

    def observe(self, reward):
        R, arm = self.action_[-1]
        self.total_reward_[R][arm] += reward
        self.pulling_times_[R][arm] += 1
        self.mean_reward_[R][arm] = self.total_reward_[R][arm] / self.pulling_times_[R][arm]
        self.t += 1

    def U(self, t, delta):
        c = 4  # the original paper doesn't specify the value of c
        conf = c * np.sqrt(np.maximum(np.log(np.maximum(np.log(t), 1) / delta), 1) / t)
        return conf

    def predict(self):
        # bracketing UCB should be an anytime algorithm
        lcb_max = -np.infty
        arm_lcb_max = 0
        for rr in range(1, self.l + 1):
            for ii in self.bracket_[rr]:
                if self.pulling_times_[rr][ii] == 0:
                    continue
                delta_ar_rr = self.delta / (len(self.bracket_[rr])) / (rr**2)
                conf = self.U(self.pulling_times_[rr][ii], delta_ar_rr)
                lcb = self.mean_reward_[rr][ii] - conf
                if lcb > lcb_max:
                    lcb_max = lcb
                    arm_lcb_max = ii
        return arm_lcb_max


class BracketingUCB_k_identification(object):
    # the algorithm doesn't take k as an input
    def __init__(self, mu0: float, K: int, delta: float = 0.1, epsilon: float = 0.1, random_seed=12345) -> None:
        """Algorithm 1, $k$-identification. We want to indentify k-arm
        whose mean reward is greater than the threshold mu0.
        The predicted arm list is $S$, and its size can be greater than k,
        as this is an anytime algorithm and it will return its prediction at each round.
        The user can terminate the algorithm at any round he likes.

        Args:
            mu0 (float): Threshold.
            K (int): Number of arms.
            delta (float, optional): Confidence level. Defaults to 0.1.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        # in the original paper, $n$ denotes the total number of arms
        # here we use $K$ to denote the numebr of arms

        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        self.K = K
        self.delta = delta
        self.mu0 = mu0
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))
        self.l = 0
        self.S = list()  # $S_0=\emptyset$

        # history
        # in bracketing UCB, the algorithm will gradually creat brakets (subset of [K]) with larger size
        # and the pulling history of each each bracket are independent
        # hence we use a dictionary to contain the pulling history of each bracket
        # the key is the index of bracket
        self.mean_reward_ = dict()
        self.total_reward_ = dict()
        self.bracket_ = dict()
        self.pulling_times_ = dict()
        self.action_ = []  # each entry is $(R_t, A_t)$

        # status
        self.l = 0
        self.t = 1
        self.R = 0

    def action(self):
        if self.t >= (2**self.l) * self.l:
            M_lp1 = np.minimum(self.K, 2 ** (self.l + 1))
            bracket = np.sort(self.random_generator.choice(np.arange(1, self.K + 1), M_lp1, replace=False))
            self.bracket_[self.l + 1] = bracket

            # create dictionary for the pulling times
            self.pulling_times_[self.l + 1] = dict()
            for arm in bracket:
                self.pulling_times_[self.l + 1][arm] = 0

            # create dictionary for the mean rewards
            self.mean_reward_[self.l + 1] = dict()
            for arm in bracket:
                self.mean_reward_[self.l + 1][arm] = 0

            # create dictionary for the total rewards
            self.total_reward_[self.l + 1] = dict()
            for arm in bracket:
                self.total_reward_[self.l + 1][arm] = 0

            self.l += 1
        # self.R = ((self.R + 1) % self.l) + 1  # this is equivalent to R=1+R*(R<l)
        self.R = 1 + self.R * (self.R < self.l)  # then R will take value 1,2,3,..,l,1,2,3,...,l,... repeatedly

        # determine the pulling arm
        for arm in [a for a in self.bracket_[self.R] if a not in self.S]:
            # check whether there exists an arm has 0 pulling times
            # once the arm enter return list, it will not be pulled
            if self.pulling_times_[self.R][arm] == 0:
                self.action_.append((self.R, arm))
                return arm
        arm = 0
        UCB_max = -np.infty
        for ii in [a for a in self.bracket_[self.R] if a not in self.S]:
            # find the arm with highest UCB
            UCB_ii = self.mean_reward_[self.R][ii] + self.U(self.pulling_times_[self.R][ii], self.delta)
            if UCB_ii > UCB_max:
                arm = ii
        self.action_.append((self.R, arm))
        return arm

    def observe(self, reward):
        R, arm = self.action_[-1]
        self.total_reward_[R][arm] += reward
        self.pulling_times_[R][arm] += 1
        self.mean_reward_[R][arm] = self.total_reward_[R][arm] / self.pulling_times_[R][arm]
        self.t += 1

        # update self.S
        # iterate p from $|A_{R_t}|$ to 1, and try to find the
        # maximum p such that $|s(p)|\geq p$
        for p in range(len(self.bracket_[R]), 0, -1):
            # calculate delta_prime_p_Art
            delta_r = self.delta / (R**2)
            delta_r_prime = delta_r / 6.4 / np.log(36 / delta_r)
            delta_prime_p_Art = delta_r_prime * p / len(self.bracket_[R])

            # construct s_p
            s_p = []
            for arm in self.bracket_[R]:
                if self.pulling_times_[R][arm] == 0:
                    continue
                conf = self.U(self.pulling_times_[R][arm], delta_prime_p_Art)
                lcb = self.mean_reward_[R][arm] - conf
                if lcb >= self.mu0:
                    s_p.append(arm)

            # update set S
            if len(s_p) >= p:
                self.S = list(set(self.S + s_p))
                break

    def U(self, t, delta):
        c = 4  # the original paper doesn't specify the value of c
        conf = c * np.sqrt(np.maximum(np.log(np.maximum(np.log(t), 1) / delta), 1) / t)
        return conf

    def if_stop(self):
        if len(self.S) == self.K:
            # all the arms are in S
            return True
        else:
            return False

    def predict(self):
        return self.S


class BracketingUCB_epsilon_good_heuristic(object):
    # the algorithm doesn't take epsilon as an input
    def __init__(self, K: int, delta: float = 0.1, start_size: int = 8, random_seed=12345) -> None:
        """Algorithm 1, $\epsilon$-good arm identification.
        In the numeric experiment part, they further implemented several heuristic tricks based on alg 1.
        1. Starting size of the bracket is 64 instead of 2. Here we take this as an input parameter.
           The variable l will record the number of existing brackets, also the index of phase.
           The size of l-th bracket will be $\min\{\text{start_size} * 2^{l-1}, K\}$.

        2. The samples are shared among different brackets.
           That means we shall not use a dictionary to separate the data collected from different brackets.

        3. The number of total bracket is finite. Once the bracket size is greater than the arm number,
           the algorithm would stop generating new bracket.

        4. To compare with LUCB, each time they will pull two arms with maximum empirical mean and
           maximizer of the upper confidence of the mean.
           At phase $l$, each bracket will be pulled $2^{l+1}$ times. The sequence of brackets to be pulled would be
           1, 1, 2, 2, ..., l, l, 1, 1, 2, 2, ..., l, l... until the pulling times of each bracket achieve $2^{l+1}$.

           In the case that the algorithm stops generating bracket, take the index of the last-generating bracket as $\bar{R}$,
           the pulling sequence of brackets will be $1, 1, 2, 2, ..., \bar{R}, \bar{R}, 1, 1, 2, 2, ..., \bar{R}, \bar{R}...$
           until the pulling times of each bracket achieve $2^{l+1}$.

           Since we may also eliminate some brackets after each round, the real bracket sequence might not be
           1, 2, 3, ..., l

        5. They will eliminate a bracket if its maximum high prob lower bound is smaller than another bracket with
           larger size.
           Here we conduct a test after each pull, to check whether we should eliminate any bracket.


        Args:
            K (int): Number of arms.
            delta (float, optional): Confidence level. Defaults to 0.1.
            start_size (int, optional): Starting size of the bracket. Defaults to be 8.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        # in the original paper, $n$ denotes the total number of arms
        # here we use $K$ to denote the numebr of arms
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        self.K = K
        self.delta = delta
        self.start_size = start_size
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))
        self.l = 0

        # history
        # in bracketing UCB, the algorithm will gradually creat brakets (subset of [K]) with larger size
        # and the pulling history of each each bracket are independent
        # hence we use a dictionary to contain the pulling history of each bracket
        # the key is the index of bracket
        self.mean_reward_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.bracket_ = dict()
        self.pulling_times_ = np.zeros(K)
        self.action_ = []  # each entry is $(R_t, A_t)$
        self.pulling_list = [(0, kk) for kk in range(1, K + 1)]

        # status
        self.l = 0
        self.t = 1
        self.R = 0
        self.R_list = []  # it will be list(self.bracket_.keys())
        self.phase_length = 0  # this variable will count the pulling times of each bracket at a phase
        self.num_bracket = 0  # this variable record the number of existing brackets

    def action(self):
        if len(self.pulling_list) > 0 and self.t <= self.K:
            # firstly we pull all the arms one time
            RR, arm = self.pulling_list.pop(0)  # all the R here would be 0
            self.action_.append((RR, arm))  # at this phase, all the arms attribute to no bracket
            return arm
        elif len(self.pulling_list) > 0 and self.t > self.K:
            RR, arm = self.pulling_list.pop(0)
            self.action_.append((RR, arm))
            return arm
        else:  # len(self.pulling_list) == 0 and self.t > self.K:
            if self.phase_length == 0:  # we need to start a new phase
                self.l += 1
                if self.start_size * (2**self.l) < self.K * 2:  # we need to create a new bracket
                    self.num_bracket += 1
                    bracket_size = np.minimum(self.start_size * (2**self.l), self.K)
                    bracket = np.sort(
                        self.random_generator.choice(np.arange(1, self.K + 1), bracket_size, replace=False)
                    )
                    self.bracket_[self.num_bracket] = bracket

                self.phase_length = 2**self.l  # setup the length of this new phase
                # find the existing minimum bracket index
                self.R_list = list(self.bracket_.keys())

            # determine the pulling arm
            self.R = self.R_list.pop(0)
            if len(self.R_list) == 0:
                self.phase_length -= 1
            bracket = self.bracket_[self.R]
            mean_reward_bracket = self.mean_reward_[bracket - 1].copy()
            high_prob_UCB = np.array(
                [self.mean_reward_[ii - 1] + self.U(self.pulling_times_[ii - 1], self.delta) for ii in bracket]
            )
            ht = bracket[np.argmax(high_prob_UCB)]
            mean_reward_bracket[ht] = -np.inf
            lt = bracket[np.argmax(mean_reward_bracket)]
            self.pulling_list.append((self.R, ht))
            self.pulling_list.append((self.R, lt))

            # return the arm to be pulled
            RR, arm = self.pulling_list.pop(0)
            self.action_.append((RR, arm))
            return arm

    def observe(self, reward):
        R, arm = self.action_[-1]
        self.total_reward_[R][arm] += reward
        self.pulling_times_[R][arm] += 1
        self.mean_reward_[R][arm] = self.total_reward_[R][arm] / self.pulling_times_[R][arm]
        self.t += 1

        # conduct a test to see whether we should eliminate some brackets
        high_prob_lcb = self.mean_reward_ - np.array(
            [self.U(self.pulling_times_[aa - 1], self.delta) for aa in range(1, self.K + 1)]
        )
        bracket_list = np.sort(list(self.bracket_.keys()))
        # use brute-force method to check whether we should eliminate any arm
        for index, rr in enumerate(bracket_list):
            high_prob_lcb_rr = high_prob_lcb[rr - 1]
            for index_ in range(index + 1, len(bracket_list)):
                high_prob_lcb_nextrr = high_prob_lcb[bracket_list[index_] - 1]
                if high_prob_lcb_nextrr > high_prob_lcb_rr:  # we should eliminate rr
                    self.bracket_.pop(rr, None)
                    self.R_list.remove(rr)

    def U(self, t, delta):
        c = 4  # the original paper doesn't specify the value of c
        conf = c * np.sqrt(np.maximum(np.log(np.maximum(np.log(t), 1) / delta), 1) / t)
        return conf

    def predict(self):
        # bracketing UCB should be an anytime algorithm
        lcb_max = -np.infty
        arm_lcb_max = 0
        for rr in range(1, self.l + 1):
            for ii in self.bracket_[rr]:
                if self.pulling_times_[rr][ii] == 0:
                    continue
                delta_ar_rr = self.delta / (len(self.bracket_[rr])) / (rr**2)
                conf = self.U(self.pulling_times_[rr][ii], delta_ar_rr)
                lcb = self.mean_reward_[rr][ii] - conf
                if lcb > lcb_max:
                    lcb_max = lcb
                    arm_lcb_max = ii
        return arm_lcb_max


# %% unit test 1, test BracketingUCB_epsilon_good
# np.random.seed(0)
# K = 4
# delta = 0.1
# epsilon = 0.1

# T = 500

# # agent = BracketingUCB_epsilon_good(K=K, delta=delta, epsilon=epsilon)
# agent = BracketingUCB_epsilon_good(K=K, delta=delta)
# predicted_arm_ = np.zeros(T)
# for tt in range(T):
#     arm = agent.action()
#     reward = arm * 0.5 + np.random.uniform(low=-0.3, high=0.3)
#     agent.observe(reward=reward)
#     predicted_arm_[tt] = agent.predict()
# print(predicted_arm_[-100:])

# %% unit test 2, test BracketingUCB_k_identification
# np.random.seed(0)
# K = 4
# delta = 0.1
# epsilon = 0.1
# mu0 = 0

# T = 5000

# # agent = BracketingUCB_epsilon_good(K=K, delta=delta, epsilon=epsilon)
# agent = BracketingUCB_k_identification(mu0=mu0, K=K, delta=delta)
# predicted_arm_ = []
# for tt in range(T):
#     arm = agent.action()
#     reward = arm * 0.5 + np.random.uniform(low=-0.3, high=0.3)
#     agent.observe(reward=reward)
#     predicted_arm_.append(agent.predict())
#     if agent.if_stop():
#         break
# print(predicted_arm_[-100:])
