import numpy as np
from scipy.optimize import linprog
from typing import Union


class OneTimeLearning(object):
    def __init__(self, m: int, n: int, b: np.array, epsilon: Union[float, np.float64] = 0.1) -> None:
        """The source code of One Time Learning algorithm.
        In the first $\epsilon n$ rounds, we always take action = 0
        we use the information collected from the first $\epsilon n$ rounds to calculate the dual variable p
        Then we take x_t(p)=1 if $pi_t> p^T a_t$, 0 if $pi_t \leq p^Ta_t$

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.array): The initial available resource.
            epsilon (Union[float, np.float64], optional): The fraction of "exploration" phase. Defaults to 0.1.
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.epsilon = epsilon

        self.pi = np.zeros(n)
        self.a = np.zeros((m, n))

        self.t = 1
        self.action_ = np.zeros(n)  # the actions are all binary value
        self.reward_ = np.zeros(n)  # record the reward in each round
        self.p = np.zeros(m)

    def action(self, pi_t, a_t):
        self.pi[self.t - 1] = pi_t
        self.a[:, self.t - 1] = a_t

        if self.t < np.ceil(self.epsilon * self.n):
            action = 0
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = 0
        elif self.t == np.ceil(self.epsilon * self.n):
            # solve the dual optimization problem to get p
            s = int(np.ceil(self.epsilon * self.n))

            c = np.ones(self.m + s)
            c[: self.m] = (1 - self.epsilon) * s / self.n * self.b
            Aub = np.zeros((s, self.m + s))
            Aub[:, : self.m] = -self.a[:, :s].T
            Aub[:, self.m : self.m + s] = -np.eye(s)
            bub = -self.pi[:s]

            res = linprog(c=c, A_ub=Aub, b_ub=bub)
            self.p = res.x[0 : self.m]

            action = (pi_t > self.p @ a_t).astype(float)
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * pi_t
        else:
            action = (pi_t > self.p @ a_t).astype(float)
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = action * pi_t

        self.t += 1
        return action


class DynamicLearning(object):
    def __init__(self, m: int, n: int, b: np.array, epsilon: Union[float, np.float64] = 0.1) -> None:
        """The source code of One Time Learning algorithm.
        In the first $\epsilon n$ rounds, we always take action = 0
        we use the information collected from the first $\epsilon n$ rounds to calculate the dual variable p
        Then we take x_t(p)=1 if $pi_t> p^T a_t$, 0 if $pi_t \leq p^Ta_t$

        Args:
            m (int): Number of resources.
            n (int): Number of rounds.
            b (np.array): The initial available resource.
            epsilon (Union[float, np.float64], optional): The fraction of "exploration" phase. Defaults to 0.1.
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.n = n
        self.b = b
        self.epsilon = epsilon

        self.pi = np.zeros(n)
        self.a = np.zeros((m, n))

        self.t = 1
        self.action_ = np.zeros(n)  # the actions are all binary value
        self.reward_ = np.zeros(n)  # record the reward in each round
        self.update_p_index = np.ceil(self.epsilon * self.n)
        self.p = np.zeros(m)

    def action(self, pi_t, a_t):
        self.pi[self.t - 1] = pi_t
        self.a[:, self.t - 1] = a_t

        if self.t < np.ceil(self.epsilon * self.n):
            action = 0
            self.action_[self.t - 1] = action
            self.reward_[self.t - 1] = 0
            self.t += 1
            return action

        if self.t == self.update_p_index:
            s = int(self.update_p_index)

            c = np.ones(self.m + s)
            c[: self.m] = (1 - self.epsilon * np.sqrt(self.n / self.update_p_index)) * self.update_p_index / self.n * self.b
            Aub = np.zeros((s, self.m + s))
            Aub[:, : self.m] = -self.a[:, :s].T
            Aub[:, self.m : self.m + s] = -np.eye(s)
            bub = -self.pi[:s]

            res = linprog(c=c, A_ub=Aub, b_ub=bub)
            self.p = res.x[0 : self.m]

            self.update_p_index *= 2  # the next moment that we update p

        action = (pi_t > self.p @ a_t).astype(float)
        self.action_[self.t - 1] = action
        self.reward_[self.t - 1] = action * pi_t

        self.t += 1
        return action


#%% unit test 1, debug OneTimeLearning
# from env import Env

# m = 3
# n = 10
# epsilon = 0.1
# random_seed = 0
# # B = 6 * m * np.log(n / epsilon) / epsilon**3
# B = 50
# b = B * np.ones(m)

# np.random.seed(random_seed)
# pi = np.random.uniform(low=0.0, high=1.0, size=(n))
# a = np.random.uniform(low=0.0, high=1.0, size=(m, n))
# opt_res = linprog(c=-pi, A_ub=a, b_ub=b, bounds=[(0.0, 1.0)] * n)
# print(f"offline optimal is {-opt_res.fun}")

# # bench mark from offline linear programming

# env = Env(m=m, n=n, b=b, pi=pi, a=a, random_seed=random_seed)
# agent = OneTimeLearning(m=m, n=n, epsilon=epsilon, b=b)
# while not env.if_stop():
#     pi_t, a_t = env.deal()
#     agent.action(pi_t=pi_t, a_t=a_t)
# print("algorithm reward is", np.sum(agent.reward_))


#%% unit test 2, test the performance of dynamic learning
# from env import Env

# m = 4
# n = 200
# epsilon = 0.1
# random_seed = 0
# # B = 6 * m * np.log(n / epsilon) / epsilon**3
# B = 40
# print(f"B is {B}")
# b = B * np.ones(m)

# np.random.seed(random_seed)
# pi = np.random.uniform(low=0.0, high=1.0, size=(n))
# a = np.random.uniform(low=0.0, high=1.0, size=(m, n))

# opt_res = linprog(c=-pi, A_ub=a, b_ub=b, bounds=[(0.0, 1.0)] * n)
# print(f"offline optimal is {-opt_res.fun}")

# env = Env(m=m, n=n, b=b, pi=pi, a=a, random_seed=random_seed)
# agent = DynamicLearning(m=m, n=n, epsilon=epsilon, b=b)
# while not env.if_stop():
#     pi_t, a_t = env.deal()
#     agent.action(pi_t=pi_t, a_t=a_t)
# print("algorithm reward is", np.sum(agent.reward_))
