from typing import Union
import numpy as np
from numpy.random import Generator, PCG64


class FALCON(object):
    def __init__(self, K: int, F: list, delta: Union[float, np.float64], c: Union[float, np.float64] = 1.0 / 30.0, T: Union[int, None] = None, random_seed=12345) -> None:
        """The algorithm FALCON assume the size of function class is known to us. Here we denote it as n.
        We will assign the epoch schedule through the parameter T.
        If T is assigned to this algorithm, we will set $\tau_m=2T^{1-2^{-m}}$.
        If T is unknown to us, we will set $\tau_m = 2^m$.
        We adopt MSE as the metric to select offline oracles.

        Args:
            K (int): Number of arms.
            F (list): List of functions.
            delta (Union[float, np.float64]): Confidence level.
            c (Union[float, np.float64]): Tuning parameter
            T (Union[int, None], optional): Number of rounds. Defaults to None.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        self.m = 1  # index of current epoch
        self.t = 1  # index of current round
        self.gamma_m = 0  # the scalar that balance exploration and exploitation
        self.hat_f_m = None  # current offline oracle

        self.K = K
        self.F = F
        self.n = len(F)  # size of function class
        assert self.n > 0, "function list is empty"
        self.delta = delta
        self.c = c
        self.T = T
        self.T_is_None = T is None
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.a_ = []  # history of action
        self.reward_ = []  # history of reward
        self.context_ = []  # history of context
        # we will clear the history at rount \tau_m

    def action(self, x_t):
        self.context_.append(x_t)

        # check whether we need to update the offline Oracle
        if self.T_is_None and self.t == 2 ** (self.m - 1):
            # $\tau_m = 2^{m}$
            # update the offline oracle
            if self.m == 1:
                self.gamma_m = 0
            else:
                tau_m_1 = 2 ** (self.m - 1)
                self.gamma_m = self.c * np.sqrt(self.K * tau_m_1 / np.log(self.n * np.log(tau_m_1) * self.m / self.delta))
            self.hat_f = self.get_function()
            self.m += 1
        elif (not self.T_is_None) and self.t == np.ceil(2 * (self.T ** (1 - 2 ** (-self.m)))):
            if self.m == 1:
                self.gamma_m = 0
            else:
                tau_m_1 = np.ceil(2 * (self.T ** (1 - 2 ** (-self.m + 1))))
                self.gamma_m = self.c * np.sqrt(self.K * tau_m_1 / np.log(self.n * np.log(tau_m_1) * self.m / self.delta))
            self.hat_f = self.get_function()
            self.m += 1

        # use current offline Oracle to select the predicted best arm
        predict_reward = np.zeros(self.K)
        for kk in range(self.K):
            predict_reward[kk] = self.hat_f(x_t, kk)
        hat_at = np.argmax(predict_reward)

        # generate probability vector to sample the action in this round
        prob_vector = 1 / (self.K + self.gamma_m * (predict_reward[hat_at] - predict_reward))
        prob_vector[hat_at] = 1 - np.sum(prob_vector[0:hat_at]) - np.sum(prob_vector[hat_at + 1 :])

        # sample action
        action = self.random_generator.choice(a=np.arange(self.K), p=prob_vector)
        self.a_.append(action)

        return action

    def observe(self, reward):
        self.reward_.append(reward)
        self.t += 1

    def get_function(self):
        if self.t == 1:
            # if history doesn't exist, we will return the first funtion in the list
            return self.F[0]

        epoch_length = len(self.a_)
        predicted_reward = np.zeros((epoch_length, self.n))
        for round_index in range(epoch_length):
            for fun_index, fun in enumerate(self.F):
                predicted_reward[round_index][fun_index] = fun(self.context_[round_index], self.a_[round_index])

        mse = np.mean((predicted_reward - np.array(self.reward_)[:, np.newaxis]) ** 2, axis=0)
        best_fun_index = np.argmin(mse)
        return self.F[best_fun_index]


class FALCONplus(object):
    def __init__(self, K: int, F: callable, E: callable, delta: Union[float, np.float64], c: Union[float, np.float64] = 1 / 2, T: Union[int, None] = None, random_seed=12345) -> None:
        """The algorithm FALCON assume the size of function class is known to us. Here we denote it as n.
        We will assign the epoch schedule through the parameter T.
        If T is assigned to this algorithm, we will set $\tau_m=2T^{1-2^{-m}}$.
        If T is unknown to us, we will set $\tau_m = 2^m$.
        We adopt MSE as the metric to select offline oracles.
        We clear the memory at the beginning of each epoch.

        Args:
            K (int): Number of arms.
            F (callable): Offline oracle that will return predict function.
            E (callable): The upper bound function of expected prediction error. Its parameters include confidence level and round number.
            delta (Union[float, np.float64]): Confidence level.
            c (Union[float, np.float64]): Tuning parameter
            T (Union[int, None], optional): Number of rounds. Defaults to None.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        self.m = 1  # index of current epoch
        self.t = 1  # index of current round
        self.gamma_m = 0  # the scalar that balance exploration and exploitation
        self.hat_f_m = None  # current offline oracle

        self.K = K
        self.F = F
        self.E = E
        assert self.n > 0, "function list is empty"
        self.delta = delta
        self.c = c
        self.T = T
        self.T_is_None = T is None
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.a_ = []  # history of action
        self.reward_ = []  # history of reward
        self.context_ = []  # history of context
        # we will clear the history at rount \tau_m

    def action(self, x_t):
        self.context_.append(x_t)

        # check whether we need to update the offline Oracle
        if self.T_is_None and self.t == 2 ** (self.m - 1):
            # $\tau_m = 2^{m}$
            # update the offline oracle
            if self.m == 1:
                self.gamma_m = 0
            else:
                tau_m_1 = 2 ** (self.m - 1)
                tau_m_2 = 2 ** (self.m - 2)
                temp_n = tau_m_1 - tau_m_2
                temp_delta = self.delta / (2 * self.m**2)
                self.gamma_m = self.c * np.sqrt(self.K / self.E(delta=temp_delta, n=temp_n))  # the upper bound might not require confidence level
            self.hat_f = self.get_function()
            self.m += 1

            self.a_ = list()
            self.context_ = list()
            self.reward_ = list()
        elif (not self.T_is_None) and self.t == np.ceil(2 * (self.T ** (1 - 2 ** (-self.m)))):
            if self.m == 1:
                self.gamma_m = 0
            else:
                tau_m_1 = np.ceil(2 * (self.T ** (1 - 2 ** (-self.m + 1))))
                tau_m_2 = np.ceil(2 * (self.T ** (1 - 2 ** (-self.m + 2))))
                temp_n = tau_m_1 - tau_m_2
                temp_delta = self.delta / (2 * self.m**2)
                self.gamma_m = self.c * np.sqrt(self.K / self.E(delta=temp_delta, n=temp_n))
            self.hat_f = self.get_function()
            self.m += 1

            self.a_ = list()
            self.context_ = list()
            self.reward_ = list()

        # use current offline Oracle to select the predicted best arm
        predict_reward = np.zeros(self.K)
        for kk in range(self.K):
            predict_reward[kk] = self.hat_f(x_t, kk)
        hat_at = np.argmax(predict_reward)

        # generate probability vector to sample the action in this round
        prob_vector = 1 / (self.K + self.gamma_m * (predict_reward[hat_at] - predict_reward))
        prob_vector[hat_at] = 1 - np.sum(prob_vector[0:hat_at]) - np.sum(prob_vector[hat_at + 1 :])

        # sample action
        action = self.random_generator.choice(a=np.arange(self.K), p=prob_vector)
        self.a_.append(action)

        return action

    def observe(self, reward):
        self.reward_.append(reward)
        self.t += 1

    def get_function(self):
        if self.t == 1:
            # if history doesn't exist, we will return constant 0 as the predicted function
            return lambda x: 0

        return self.F(self.context_, self.a_, self.reward_)


#%% unit test 1, test FALCON:get_function
# def fun_sin(x, k):
#     return (np.sin(x[k]) + 1) / 2


# def fun_sin_2(x, k):
#     return (np.sin(x[k] ** 2) + 1) / 2


# def fun_cos(x, k):
#     return (np.cos(x[k]) + 1) / 2


# def fun_cos_2(x, k):
#     return (np.cos(x[k] ** 2) + 1) / 2


# F = [fun_sin, fun_sin_2, fun_cos, fun_cos_2]
# K = 4
# d = K
# epoch_length = 100
# random_seed = 12345
# np.random.seed(random_seed)
# agent = FALCON(K=K, F=F, delta=0.1, c=1.0, T=None)
# agent.a_ = list(np.random.randint(low=0, high=K, size=epoch_length))
# for epoch_index in range(epoch_length):
#     agent.context_.append(np.random.uniform(low=0.0, high=10.0, size=d))
# for epoch_index in range(epoch_length):
#     agent.reward_.append(F[3](agent.context_[epoch_index], agent.a_[epoch_index]) + np.random.normal(loc=0.0, scale=0.5))
# fun = agent.get_function()
# print(fun.__name__)

#%% unit test 2, test the loop including class agent FALCON and env
# from env import Env
# import matplotlib.pyplot as plt


# def fun_sin(x, k):
#     return (np.sin(x[k]) + 1) / 2


# def fun_sin_2(x, k):
#     return (np.sin(x[k] ** 2) + 1) / 2


# def fun_cos(x, k):
#     return (np.cos(x[k]) + 1) / 2


# def fun_cos_2(x, k):
#     return (np.cos(x[k] ** 2) + 1) / 2


# F = [fun_sin, fun_sin_2, fun_cos, fun_cos_2]
# K = 4
# d = K
# delta = 0.1
# c = 1

# agent = FALCON(K=K, F=F, delta=delta, c=c)
# env = Env(K=K, d=d, f_real=F[2])

# T = 100
# best_reward = np.zeros(T)
# agent_reward = np.zeros(T)
# for tt in range(T):
#     context = env.deal()
#     action = agent.action(context)
#     reward = env.response(action)
#     agent.observe(reward)

#     temp_reward = np.zeros(K)
#     for kk in range(K):
#         temp_reward[kk] = env.f_real(context, kk)
#     best_reward[tt] = np.max(temp_reward)

#     agent_reward[tt] = env.f_real(context, action)

# plt.plot(np.cumsum(agent_reward), label="agent")
# plt.plot(np.cumsum(best_reward), label="best")
# plt.legend()
# plt.show()

#%% unit test 3, test the class agent FALCONplus and env
def Flinear(context, action, reward, d=4):
    # context.shape = [epoch_num, K*d]
    temp_context = np.zeros((context.shape[0], d))
    for row_index in range(context.shape[0]):
        temp_context[row_index, :] = context[row_index, int(action[row_index] * d) : int(action[row_index] + 1) * d]

    theta = np.linalg.solve(temp_context.T @ temp_context, temp_context.T @ reward)

    def hat_f(context, action):
        return context[action * d : (action + 1) * d] @ theta

    return hat_f


def E(d=4, **kwargs):
    return np.sqrt(d / kwargs["n"])


# np.random.seed(12345)
# d = 2
# K = 2
# epoch_num = 10

# context = np.random.uniform(size=(epoch_num, K * d))

# action = np.zeros(epoch_num)
# action[epoch_num // 2 :] = 1

# theta = np.ones(d)

# reward = np.zeros(epoch_num)
# reward[0 : epoch_num // 2] = context[0 : epoch_num // 2, 0:2] @ theta
# reward[epoch_num // 2 :] = context[epoch_num // 2 :, 2:] @ theta

# hat_f = Flinear(context, action, reward, d=d)
# context = np.random.uniform(size=(5, K * d))
# print(context)
# for epoch_index in range(5):
#     print(hat_f(context[epoch_index, :], action=1), "correct answer", np.sum(context[epoch_index, 2:]))


#%%
# def E(**kwargs):
#     for key in kwargs.keys():
#         print(key, kwargs[key])
