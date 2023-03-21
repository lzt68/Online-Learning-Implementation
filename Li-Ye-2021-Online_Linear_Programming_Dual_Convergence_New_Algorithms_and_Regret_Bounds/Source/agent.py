import numpy as np
from scipy.optimize import linprog
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
        self.input_name = input_name

        np.random.seed(random_seed)
        if input_name == "RandomInputI":
            self.p = np.zeros(m)  # please check section Note in README.md to see why we can assert best solution is 0 vector
        elif input_name == "RandomInputII":
            n_sample = 1000000
            sample_size = 10000
            epoch_num = 20

            step_size = 1.0

            a = np.random.uniform(low=-0.5, high=1.0, size=(m, n_sample))
            r = np.sum(a, axis=0)
            d = 0.2

            p0 = 1.0
            pnext = 1.0
            for _ in range(epoch_num):
                index = np.arange(n_sample)
                np.random.shuffle(index)
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
        print("Successfully get p")

        self.pi = np.zeros(n)
        self.a = np.zeros((m, n))
        self.action_ = np.zeros(n)
        self.reward_ = np.zeros(n)

        self.t = 1

    def action(self, r_t, a_t):
        self.pi[self.t - 1] = r_t
        self.a[:, self.t - 1] = a_t

        action = (r_t > self.p @ a_t).astype(float)
        self.action_[self.t - 1] = action
        self.reward_[self.t - 1] = action * r_t

        self.t += 1
        return action


#%% unit test 1, debug no-need-to-learn
# m = 4
# n = 25
# b = 0.25 * np.ones(m) * n
# agent = NoNeedtoLearn(m=m, n=n, b=b, random_seed=2)
# print(agent.p)

#%% unit test 2, debug OneTimeLearning
# from env import RandomInputI

# m = 4
# n = 25
# d = 0.25
# b = d * np.ones(m) * n
# epsilon = 0.1
# random_seed = 0

# # bench mark from offline linear programming
# env = RandomInputI(m=m, n=n, b=b, random_seed=random_seed)
# agent = NoNeedtoLearn(m=m, n=n, b=b)
# while not env.if_stop():
#     r_t, a_t = env.deal()
#     action = agent.action(r_t=r_t, a_t=a_t)
#     env.observe(action)
# print("algorithm reward is", np.sum(agent.reward_))

#%% unit test 2, test the performance of dynamic learning
# from env import Env

# m = 4
# n = 100
# epsilon = 0.1
# random_seed = 0
# B = 10
# print(f"B is {B}")
# b = B * np.ones(m)

# np.random.seed(random_seed)
# pi = np.random.uniform(low=0.0, high=1.0, size=(n))
# a = np.random.uniform(low=0.0, high=1.0, size=(m, n))

# opt_res = linprog(c=-pi, A_ub=a, b_ub=b, bounds=[(0.0, 1.0)] * n)
# print(f"offline optimal is {-opt_res.fun}")

# env = Env(m=m, n=n, b=b, pi=pi, a=a, random_seed=random_seed)
# agent = OneTimeLearning(m=m, n=n, epsilon=epsilon, b=b)
# while not env.if_stop():
#     pi_t, a_t = env.deal()
#     action = agent.action(pi_t=pi_t, a_t=a_t)
#     env.observe(action)
# print("one time learning, algorithm reward is", np.sum(agent.reward_))

# env = Env(m=m, n=n, b=b, pi=pi, a=a, random_seed=random_seed)
# agent = DynamicLearning(m=m, n=n, epsilon=epsilon, b=b)
# while not env.if_stop():
#     pi_t, a_t = env.deal()
#     action = agent.action(pi_t=pi_t, a_t=a_t)
#     env.observe(action)
# print("Dynamic learning, algorithm reward is", np.sum(agent.reward_))
