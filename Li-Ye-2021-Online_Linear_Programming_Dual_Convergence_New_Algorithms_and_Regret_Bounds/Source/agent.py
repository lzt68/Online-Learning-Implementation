import numpy as np
from scipy.optimize import linprog
from numpy.random import Generator, PCG64
from typing import Union


class NoNeedtoLearn(object):
    def __init__(self, m: int, n: int, b: np.array, input_name: str = "RandomInputI", random_seed: int = 12345) -> None:
        """Assume the distribution of (r, a) is known

        Solve the stochastic programming problem
        $$
        p* = \arg\min_{p\ge 0} \ d^Tp +\mathbb{E}_{(r,\bold{a})\sim \mathcal{P}}\left[(r-\bold{a}^Tp)^+\right]
        $$
        We solve this stochastic programming problem via SAA scheme with 1000 samples.
        Then we take x_t=1 if $r_t> a_t^T p^*$, 0 if $r_t \leq a_t^T p^*$.

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.array): The initial available resource.
            Input_name (str, optional): The name of the distribution. Defaults to "RandomInputI".
            random_seed (int, optional): The random seed for generating random number
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.remain_b = self.b.copy()
        self.random_generator = Generator(PCG64(random_seed))
        self.input_name = input_name

        np.random.seed(random_seed)
        if input_name == "RandomInputI":
            self.p = np.zeros(m)  # please check section Note in README.md to see why we can assert best solution is 0 vector
        elif input_name == "RandomInputII":
            n_sample = 1000000
            sample_size = 10000
            epoch_num = 20

            step_size = 1.0

            a = self.random_generator.uniform(low=0.5, high=1.0, size=(m, n_sample))
            r = np.sum(a, axis=0)
            d = 0.2

            p0 = 1.0
            pnext = 1.0
            for _ in range(epoch_num):
                index = np.arange(n_sample)
                self.random_generator.shuffle(index)
                a = a[:, index]
                r = r[index]
                for round_index in range(n_sample // sample_size):
                    p0 = pnext

                    a_sub = a[:, round_index * sample_size : (round_index + 1) * sample_size]
                    r_sub = r[round_index * sample_size : (round_index + 1) * sample_size]

                    y = np.ones(sample_size)
                    y[r_sub - p0 * np.sum(a_sub, axis=0) <= 0] = 0
                    y = y[np.newaxis, :]
                    grad_p = m * d * sample_size / n_sample + np.sum(y * (-np.sum(a_sub, axis=0))) / n_sample

                    pnext = p0 - grad_p * step_size
                    pnext = np.maximum(pnext, 0)
            self.p = p0 * np.ones(m)
        else:
            assert False, "Cannot find the distribution of (r, a)"
        # print("NoNeedtoLearn Successfully get p")

        self.pi = np.zeros(n)
        self.a = np.zeros((m, n))
        self.action_ = np.zeros(n)
        self.reward_ = np.zeros(n)

        self.t = 1

    def action(self, r_t, a_t):
        self.pi[self.t - 1] = r_t
        self.a[:, self.t - 1] = a_t

        if np.all(a_t <= self.remain_b):
            action = (r_t > self.p @ a_t).astype(float)
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * r_t
        else:
            action = 0
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * r_t
        self.remain_b = self.remain_b - a_t * action

        self.t += 1
        return action


class SimplifiedDynamicLearning(object):
    def __init__(self, m: int, n: int, b: np.array) -> None:
        """The price vector $p_t$ is updated only at geometric time intervals and are independent of actions

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.array): The initial available resource.
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.remain_b = self.b.copy()
        self.d = b / n

        self.L = int(np.ceil(np.log2(n)))  # check README.md to see why we take this value
        self.delta = n ** (1 / self.L)

        self.pi = np.zeros(n)
        self.a = np.zeros((m, n))
        self.p = np.zeros(m)
        self.action_ = np.zeros(n)
        self.reward_ = np.zeros(n)

        self.t = 1
        self.k = 1

    def action(self, r_t, a_t):
        self.pi[self.t - 1] = r_t
        self.a[:, self.t - 1] = a_t

        if self.t == int(np.floor(self.delta**self.k)):
            self.updatep()
            self.k += 1

        if np.all(a_t <= self.remain_b):
            action = (r_t > self.p @ a_t).astype(float)
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * r_t
        else:
            action = 0
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * r_t
        self.remain_b = self.remain_b - a_t * action

        self.t += 1
        return action

    def updatep(self):
        # solve the dual problem
        # $$
        # p^*_k=\arg\min_p \sum_{i=1}^m d_ip_i+\frac{1}{t_k}\sum_{j=1}^{t_k}\left(r_j-\sum_{i=1}^m a_{ij} p_i\right)^+, s.t. p\ge0
        # $$
        c = np.ones(self.m + self.t) / self.t
        c[: self.m] = self.d.copy()

        Aub = np.zeros((self.t, self.m + self.t))
        Aub[:, : self.m] = -self.a[:, : self.t].T
        Aub[:, self.m : self.m + self.t] = -np.eye(self.t)
        bub = -self.pi[: self.t]

        res = linprog(c=c, A_ub=Aub, b_ub=bub)
        self.p = res.x[: self.m]


class ActionHistoryDependentLearning(object):
    def __init__(self, m: int, n: int, b: np.array) -> None:
        """The price vector $p_t$ is dependent on remaining resources

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.array): The initial available resource.
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.remain_b = self.b.copy()
        self.d = b / n

        self.pi = np.zeros(n)
        self.a = np.zeros((m, n))
        self.p = np.zeros(m)
        self.action_ = np.zeros(n)
        self.reward_ = np.zeros(n)

        self.t = 1
        self.k = 0

    def action(self, r_t, a_t):
        self.pi[self.t - 1] = r_t
        self.a[:, self.t - 1] = a_t

        if np.all(a_t <= self.remain_b):
            action = (r_t > self.p @ a_t).astype(float)
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * r_t
        else:
            action = 0
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * r_t
        self.remain_b = self.remain_b - a_t * action

        # update p
        self.updatep()
        self.t += 1
        return action

    def updatep(self):
        # solve the dual problem
        # $$
        # p^*_k=\arg\min_p \sum_{i=1}^m d_ip_i+\frac{1}{t_k}\sum_{j=1}^{t_k}\left(r_j-\sum_{i=1}^m a_{ij} p_i\right)^+, s.t. p\ge0
        # $$
        if self.t == self.n:
            return

        c = np.ones(self.m + self.t) / self.t
        c[: self.m] = self.remain_b / (self.n - self.t)

        Aub = np.zeros((self.t, self.m + self.t))
        Aub[:, : self.m] = -self.a[:, : self.t].T
        Aub[:, self.m : self.m + self.t] = -np.eye(self.t)
        bub = -self.pi[: self.t]

        res = linprog(c=c, A_ub=Aub, b_ub=bub)
        self.p = res.x[: self.m]


#%% unit test 1, debug no-need-to-learn
# m = 4
# n = 25
# b = 0.25 * np.ones(m) * n
# agent = NoNeedtoLearn(m=m, n=n, b=b, random_seed=2)
# print(agent.p)

#%% unit test 2, debug OneTimeLearning
# from env import RandomInputI

# m = 4
# n = 100
# d = 0.25
# b = d * np.ones(m) * n
# random_seed = 0

# env = RandomInputI(m=m, n=n, b=b, random_seed=random_seed)
# agent = NoNeedtoLearn(m=m, n=n, b=b, input_name="RandomInputI")
# while not env.if_stop():
#     r_t, a_t = env.deal()
#     action = agent.action(r_t=r_t, a_t=a_t)
#     env.observe(action)
# print(agent.p)
# print("OneTimeLearning algorithm reward is", np.sum(agent.reward_))

#%% unit test 3, debug SimplifiedDynamicLearning
# from env import RandomInputI

# m = 4
# n = 100
# d = 0.25
# b = d * np.ones(m) * n
# random_seed = 0

# env = RandomInputI(m=m, n=n, b=b, random_seed=random_seed)
# agent = SimplifiedDynamicLearning(m=m, n=n, b=b)
# while not env.if_stop():
#     r_t, a_t = env.deal()
#     action = agent.action(r_t=r_t, a_t=a_t)
#     env.observe(action)
# print("SimplifiedDynamicLearning algorithm reward is", np.sum(agent.reward_))

#%% unit test 4, debug ActionHistoryDependentLearning
# from env import RandomInputI
# from time import time

# m = 4
# n = 500
# d = 0.25
# b = d * np.ones(m) * n
# random_seed = 0

# # bench mark from offline linear programming
# env = RandomInputI(m=m, n=n, b=b, random_seed=random_seed)
# agent = ActionHistoryDependentLearning(m=m, n=n, b=b)
# t1 = time()
# while not env.if_stop():
#     r_t, a_t = env.deal()
#     action = agent.action(r_t=r_t, a_t=a_t)
#     env.observe(action)
# t2 = time()
# print(f"ActionHistoryDependentLearning algorithm reward is {np.sum(agent.reward_)}, time consumption is {t2-t1}")

# %%
