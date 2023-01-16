import numpy as np
import random
from copy import deepcopy
from GameSetting import *
from NeuralNetworkRelatedFunction import *
import torch


class NeuralAgent(torch.nn.module):
    def __init__(self, K: int, T: int, d: int, L: int = 2, m: int = 20, v: float = 0.1, lambda_: float = 0.01, eta: float = 0.001, frequency: int = 50, batchsize: int = None, verbose: bool = True):
        """_summary_

        Args:
            K (int): Total number of actions
            T (int): Total number of periods
            d (int): The dimension of context
            L (int, optional): The number of hidden layer. Defaults to 2.
            m (int, optional): The number of neurals in each layer. Defaults to 20.
            v (float, optional): _description_. Defaults to 0.1.
            lambda_ (float, optional): _description_. Defaults to 0.01.
            eta (float, optional): _description_. Defaults to 0.001.
            frequency (int, optional): The interval between two training epoches. Defaults to 50.
            batchsize (int, optional): The size of sample batch in SGD. Defaults to None.
            verbose (bool, optional): Control whether print the traning detail. Defaults to True.
        """
        pass


class NeuralAgent_numpy:
    def __init__(self, K, T, d, L=2, m=20, v=0.1, lambda_=0.01, eta=0.001, frequency=50, batchsize=None, verbose=True):
        # K is Total number of actions,
        # T is Total number of periods
        # d is the dimension of context
        # L is the number of hidden layer
        # m is the number of neurals in each layer
        # the definition of v, lambda_, eta can be found on the original paper
        # frequency is the gap between neighbour training epoch
        # batchsize is the size of sample batch in SGD
        # verbose control whether print the traning detail
        self.K = K
        self.T = T
        self.d = d

        self.L = L
        self.m = m
        self.v = v
        self.lambda_ = lambda_
        self.eta = eta
        self.frequency = frequency  # we train the network after frequency, e.g. per 50 round
        self.batchsize = batchsize
        self.verbose = verbose
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.predicted_reward = np.zeros(T)
        self.history_context = np.zeros((d, T))

        # initialize the value of parameter
        np.random.seed(12345)
        self.theta_0 = {}
        W = np.random.normal(loc=0, scale=4 / m, size=(int(m / 2), int(m / 2)))
        w = np.random.normal(loc=0, scale=2 / m, size=(1, int(m / 2)))
        for key in range(1, L + 1):
            if key == 1:
                # this paper doesn't present the initialization of w1
                # in its setting, d = m, then he let theta_0["w1"]=[W,0;0,W]
                # but in fact d might not equal to m
                tempW = np.random.normal(loc=0, scale=4 / m, size=(int(m / 2), int(d / 2)))
                self.theta_0["w1"] = np.zeros((m, d))
                self.theta_0["w1"][0 : int(m / 2), 0 : int(d / 2)] = tempW
                self.theta_0["w1"][int(m / 2) :, int(d / 2) :] = tempW
            elif 2 <= key and key <= L - 1:
                self.theta_0["w" + str(key)] = np.zeros((m, m))
                self.theta_0["w" + str(key)][0 : int(m / 2), 0 : int(m / 2)] = W
                self.theta_0["w" + str(key)][int(m / 2) :, int(m / 2) :] = W
            else:
                self.theta_0["w" + str(key)] = np.concatenate([w, -w], axis=1)

        self.p = m + m * d + m * m * (L - 2)
        self.params = deepcopy(self.theta_0)
        self.U = lambda_ * np.eye(self.p)
        self.params_history = {}
        self.grad_history = {}

    def Action(self, context_list):
        # context_list is a d*K matrix, each column represent a context
        # the return value is the action we choose, represent the index of action, is a scalar
        sample_estimated_reward = np.zeros(self.K)  # the upper bound of K actions
        predict_reward = np.zeros(self.K)
        U_inverse = np.linalg.inv(self.U)
        for a in range(1, self.K + 1):
            predict_reward[a - 1] = NeuralNetwork(context_list[:, a - 1], self.params, self.L, self.m)["x" + str(self.L)][0]
            grad_parameter = GradientNeuralNetwork(context_list[:, a - 1], self.params, self.L, self.m)
            grad_parameter = FlattenDict(grad_parameter, self.L)
            sigma_square = self.lambda_ * grad_parameter.dot(U_inverse).dot(grad_parameter) / self.m
            sigma = np.sqrt(sigma_square)

            sample_estimated_reward[a - 1] = np.random.normal(loc=predict_reward[a - 1], scale=self.v * sigma)

        ind = np.argmax(sample_estimated_reward, axis=None)
        self.predicted_reward[self.t] = sample_estimated_reward[ind]
        self.history_action[self.t] = ind
        self.history_context[:, self.t] = context_list[:, ind]
        return ind

    def Update(self, reward):
        # reward is the realized reward after we adopt policy, a scalar
        #         print("round {:d}".format(self.t))
        self.history_reward[self.t] = reward
        ind = self.history_action[self.t]
        context = self.history_context[:, self.t]

        # compute Z_t_minus1
        grad_parameter = GradientNeuralNetwork(context, self.params, self.L, self.m)
        grad_parameter = FlattenDict(grad_parameter, self.L)
        grad_parameter = np.expand_dims(grad_parameter, axis=1)
        self.U = self.U + grad_parameter.dot(grad_parameter.transpose()) / self.m

        # train neural network
        if self.t % self.frequency == 0 and self.t > 0:
            J = self.t
        else:
            J = 0

        if self.batchsize == None:
            trainindex = range(0, self.t + 1)
        else:
            if self.batchsize > self.t + 1:
                trainindex = range(0, self.t + 1)
            else:
                trainindex = random.sample(range(0, self.t + 1), self.batchsize)

        grad_loss = {}
        for j in range(J):

            grad_loss = GradientLossFunction(
                self.history_context[:, trainindex], self.params, self.L, self.m, self.history_reward[trainindex], self.theta_0, self.lambda_  # we had not update self.t yet, so here we must +1
            )
            #             if j < 10:
            #                 eta = 1e-4
            #             else:
            #                 eta = self.eta
            eta = self.eta
            for key in self.params.keys():
                self.params[key] = self.params[key] - eta * grad_loss[key]
            loss = LossFunction(self.history_context[:, trainindex], self.params, self.L, self.m, self.history_reward[trainindex], self.theta_0, self.lambda_)
        #             print("j {:d}, loss {:4f}".format(j, loss))
        if self.verbose:
            print("round {:d}, predicted reward {:4f}, actual reward {:4f}".format(self.t, self.predicted_reward[self.t], reward))

        self.params_history[self.t] = deepcopy(self.params)
        self.grad_history[self.t] = deepcopy(grad_loss)

        self.t = self.t + 1


class BestAgent:
    def __init__(self, K, T, d, A):
        # K is Total number of actions,
        # T is Total number of periods
        # d is the dimension of context
        self.K = K
        self.T = T
        self.d = d
        self.A = A
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.history_context = np.zeros((d, T))

    def Action(self, context_list):
        # context_list is a d*K matrix, each column represent a context
        # the return value is the action we choose, represent the index of action, is a scalar

        expected_reward = np.zeros(self.K)
        for kk in range(0, self.K):
            context = context_list[:, kk]
            innerproduct = self.A.dot(context)
            expected_reward[kk] = 2 * np.exp(innerproduct) / (1 + np.exp(innerproduct))
        ind = np.argmax(expected_reward, axis=None)
        self.history_context[:, self.t] = context_list[:, ind]
        self.history_action[self.t] = ind
        return ind

    def Update(self, reward):
        # reward is the realized reward after we adopt policy, a scalar
        self.history_reward[self.t] = reward
        self.t = self.t + 1


class UniformAgent:
    def __init__(self, K, T, d):
        # K is Total number of actions,
        # T is Total number of periods
        # d is the dimension of context
        self.K = K
        self.T = T
        self.d = d
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.history_context = np.zeros((d, T))

    def Action(self, context_list):
        # context_list is a d*K matrix, each column represent a context
        # the return value is the action we choose, represent the index of action, is a scalar

        ind = np.random.randint(0, high=self.K)  # we just uniformly choose an action
        self.history_context[:, self.t] = context_list[:, ind]
        return ind

    def Update(self, reward):
        # reward is the realized reward after we adopt policy, a scalar
        self.history_reward[self.t] = reward
        self.t = self.t + 1

    def GetHistoryReward(self):
        return self.history_reward

    def GetHistoryAction(self):
        return self.history_action

    def GetHistoryContext(self):
        return self.history_context
