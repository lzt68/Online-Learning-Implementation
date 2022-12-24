import numpy as np
import torch
import random
from copy import deepcopy


class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        d: int,
        L: int = 2,
        m: int = 20,
        random_seed: int = 12345,
        device: torch.device = torch.device("cpu"),
    ):
        """The proposed neural network structure in Zhou 2020

        Args:
            d (int): Dimension of input layer.
            L (int, optional): Number of Layers. Defaults to 2.
            m (int, optional): Width of each layer. Defaults to 20.
            random_seed (int, optional): rando_seed. Defaults to 12345.
            device (torch.device, optional): The device of calculateing tensor. Defaults to torch.device("cpu").
        """
        super().__init__()
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.d = d
        self.L = L
        self.m = m
        self.random_seed = random_seed
        self.activation = torch.nn.ReLU()

        self.device = device
        print(f"Using device {self.device}")

        self.W = torch.nn.ParameterDict()
        w_for_1 = np.random.randn(d // 2, m // 2) * np.sqrt(4 / m)
        w_for_1_to_Lminus1 = np.random.randn(m // 2, m // 2) * np.sqrt(4 / m)
        w_for_L = np.random.randn(m // 2) * np.sqrt(2 / m)
        for layer_index in range(1, L + 1):
            if layer_index == 1:
                W = np.zeros((d, m))
                W[0 : d // 2, 0 : m // 2] = w_for_1
                W[d // 2 :, m // 2 :] = w_for_1
                self.W["W1"] = torch.nn.Parameter(torch.from_numpy(W)).to(self.device)
            elif layer_index == L:
                W = np.zeros((m, 1))
                W[0 : m // 2, 0] = w_for_L
                W[m // 2 :, 0] = -w_for_L
                self.W[f"W{layer_index}"] = torch.nn.Parameter(torch.from_numpy(W)).to(self.device)
            else:
                W = np.zeros((m, m))
                W[0 : m // 2, 0 : m // 2] = w_for_1_to_Lminus1
                W[m // 2 :, m // 2 :] = w_for_1_to_Lminus1
                self.W[f"W{layer_index}"] = torch.nn.Parameter(torch.from_numpy(W)).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """we accept a Tensor of input data and we must return
        a Tensor of output data

        Args:
            x (torch.Tensor): The observed context of each arm

        Returns:
            torch.Tensor: The predicted mean reward of each arm
        """
        assert x.shape[1] == self.d, "Dimension doesn't match"
        x = x.to(self.device)
        for layer_index in range(1, self.L + 1):
            x = torch.matmul(x, self.W[f"W{layer_index}"])
            if layer_index != self.L:
                x = self.activation(x)
        x = x * np.sqrt(self.m)
        return x

    def GetGrad(self, x: torch.tensor) -> np.ndarray:
        """Given the vector of context, return the flattern gradient of parameter

        Args:
            x (torch.tensor): x.shape = (d,)

        Returns:
            np.ndarray: The gradient of parameter at given point
        """
        x = x[None, :]  # expand the dimension of x
        output = self.forward(x)[0, 0]
        output.backward()

        grad = np.array([])
        for para in self.parameters():
            grad = np.concatenate([grad, para.grad.cpu().detach().numpy().flatten()], axis=0)
        return grad


class BestAgent:
    def __init__(self, K, T, d, A):
        # K is Total number of actions,
        # T is Total number of periods
        # d is the dimension of context
        # A is the context
        self.K = K
        self.T = T
        self.d = d
        self.t = 0  # marks the index of period
        self.A = A
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.history_context = np.zeros((d, T))

    def Action(self, context_list):
        # context_list is a d*K matrix, each column represent a context
        # the return value is the action we choose, represent the index of action, is a scalar

        expected_reward = np.zeros(self.K)
        for kk in range(0, self.K):
            context = context_list[kk, :]
            expected_reward[kk] = context.transpose().dot(self.A.transpose().dot(self.A)).dot(context)
        ind = np.argmax(expected_reward, axis=None)
        self.history_context[:, self.t] = context_list[ind, :]
        self.history_action[self.t] = ind
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
        self.history_context[:, self.t] = context_list[ind, :]
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


class NeuralAgent:
    def __init__(
        self,
        K: int,
        T: int,
        d: int,
        L: int = 2,
        m: int = 20,
        gamma_t: float = 0.01,
        v: float = 0.1,
        lambda_: float = 0.01,
        delta: float = 0.01,
        S: float = 0.01,
        eta: float = 0.001,
        frequency: int = 50,
        batchsize: int = 50,
        verbose: bool = True,
    ):
        """The proposed Neural UCB algorithm for solving contextual bandits

        Args:
            K (int): Number of arms
            T (int): Number of rounds
            d (int): Dimension of context
            L (int, optional): Number of Layers. Defaults to 2.
            m (int, optional): Width of each layer. Defaults to 20.
            gamma_t (float, optional): Exploration parameter. Defaults to 0.01.
            v (float, optional): Exploration parameter. Defaults to 0.1.
            lambda_ (float, optional): Regularization parameter. Defaults to 0.01.
            delta (float, optional): Confidence parameter. Defaults to 0.01.
            S (float, optional): Norm parameter. Defaults to 0.01.
            eta (float, optional): Step size. Defaults to 0.001.
            frequency (int, optional): The interval between two training rounds. Defaults to 50.
            batchsize (int, optional): The batchsize of applying SGD on the neural network. Defaults to None.
            verbose (bool, optional): Whether we will output the training process. Defaults to True.
        """
        self.K = K
        self.T = T
        self.d = d

        self.L = L
        self.m = m
        self.gamma_t = gamma_t
        self.v = v
        self.lambda_ = lambda_
        self.delta = delta
        self.S = S
        self.eta = eta
        self.frequency = frequency  # we train the network after frequency, e.g. per 50 round
        self.batchsize = batchsize
        self.verbose = verbose
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.predicted_reward = np.zeros(T)
        self.predicted_reward_upperbound = np.zeros(T)
        self.history_context = np.zeros((T, d))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mynn = NeuralNetwork(d=d, L=L, m=m, device=self.device)
        self.optimizer = torch.optim.SGD(self.mynn.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.p = m + m * d + m * m * (L - 2)
        self.Z_t_minus1 = lambda_ * np.eye(self.p)

    def Action(self, context_list: np.array) -> int:
        """Given the observed context of each arm, return the predicted arm

        Args:
            context_list (np.array): The observed context of each arm. context_list.shape = (K, d)

        Returns:
            int: the index of predicted arm, take value from 0, 1, ..., K-1
        """
        predict_reward = self.mynn.forward(torch.from_numpy(context_list))[:, 0]
        predict_reward = predict_reward.cpu().detach().numpy()

        Z_t_minus1_inverse = np.linalg.inv(self.Z_t_minus1)

        confidence = np.zeros(self.K)
        for arm in range(1, self.K + 1):
            grad_arm = self.mynn.GetGrad(torch.from_numpy(context_list[arm - 1, :]))
            confidence[arm - 1] = np.sqrt(grad_arm.dot(Z_t_minus1_inverse).dot(grad_arm) / self.m)

        # calculate the upper confidence bound
        ucb = predict_reward + self.gamma_t * confidence
        ind = np.argmax(ucb)

        # save the history
        self.history_action[self.t] = ind
        self.history_context[self.t, :] = context_list[ind, :]
        self.predicted_reward[self.t] = predict_reward[ind]
        self.predicted_reward_upperbound = ucb[ind]
        return ind

    def Update(self, reward):
        self.history_reward[self.t] = reward
        ind = self.history_action[self.t]
        context = self.history_context[self.t, :]

        # compute Z_t_minus1
        grad_parameter = self.mynn.GetGrad(torch.from_numpy(context))
        grad_parameter = np.expand_dims(grad_parameter, axis=1)
        self.Z_t_minus1 = self.Z_t_minus1 + grad_parameter.dot(grad_parameter.transpose()) / self.m

        if (self.t + 1) % self.frequency == 0:  # train the network
            # shuffle the history and conduct SGD
            history_index = np.arange(self.t + 1)
            np.random.shuffle(history_index)
            temp_history_context = self.history_context[history_index, :]
            temp_history_reward = self.history_reward[history_index]
            for batch_index in range(0, self.t // self.batchsize + 1):
                # split the batch
                if batch_index < self.t // self.batchsize:
                    X_temp = torch.from_numpy(temp_history_context[batch_index * self.batchsize : (batch_index + 1) * self.batchsize, :]).to(self.device)
                    y_temp = torch.from_numpy(temp_history_reward[batch_index * self.batchsize : (batch_index + 1) * self.batchsize]).to(self.device)
                else:
                    X_temp = torch.from_numpy(temp_history_context[batch_index * self.batchsize :, :]).to(self.device)
                    y_temp = torch.from_numpy(temp_history_reward[batch_index * self.batchsize :]).to(self.device)

                # update the neural network
                self.optimizer.zero_grad()
                output = self.mynn.forward(X_temp)
                loss = self.criterion(output[:, 0], y_temp)
                loss.backward()
                self.optimizer.step()

        self.t += 1


#%% unit test 1 -- test customized network
# d = 4
# L = 2
# m = 4
# mynn = NeuralNetwork(d=d, L=L, m=m)
# for para in mynn.parameters():
#     print(para)
# batchsize = 4
# x = np.zeros((batchsize, d))
# x = torch.from_numpy(x)
# y = mynn.forward(x)
# print(y)
# x = np.random.normal(loc=0.0, scale=1.0, size=(batchsize, d // 2))
# x = np.concatenate([x, x], axis=1)
# x = torch.from_numpy(x)
# y = mynn.forward(x)
# print(y)


#%% unit test 2 -- test customized network
# from sklearn.model_selection import train_test_split

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# d = 4
# L = 2
# m = 4
# mynn = NeuralNetwork(d=d, L=L, m=m, device=device)
# mynn.to(device=device)

# batchsize = 10
# x = np.random.normal(loc=0.0, scale=1.0, size=(batchsize, d // 2))
# x = np.concatenate([x, x], axis=1)
# A = np.random.uniform(low=0.0, high=1.0, size=(d, d))
# A = A + A.T
# # y = np.diag(x @ A @ x.T)
# y = np.sum(x @ A, axis=1)

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# X_train = torch.from_numpy(X_train)
# X_train = X_train.to(device)
# X_test = torch.from_numpy(X_test)
# X_test = X_test.to(device)
# y_train = torch.from_numpy(y_train)
# y_train = y_train.to(device)
# y_test = torch.from_numpy(y_test)
# y_test = y_test.to(device)

# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(mynn.parameters(), lr=0.01, momentum=0.9)
# for train_index in range(5000):
#     optimizer.zero_grad()
#     # output = mynn.forward(X_train)[:, 0]
#     # loss = criterion(output, y_train)
#     output = mynn.forward(X_train)
#     loss = criterion(output[:, 0], y_train)
#     loss.backward()
#     optimizer.step()

#     if (train_index + 1) % 50 == 0:
#         output_test = mynn.forward(X_test)[:, 0]
#         loss_test = criterion(output_test, y_test)
#         print(f"round {train_index+1}, train MSE {loss.cpu().detach().numpy()}, test MSE {loss_test.cpu().detach().numpy()}")

# grad = mynn.GetGrad(X_train[0, :])
# print(grad)

#%% unit test 3 -- test customized network
# d = 4
# L = 2
# m = 4
# np.random.seed(12345)

# Weight = dict()
# w_for_1 = np.random.randn(d // 2, m // 2) * np.sqrt(4 / m)
# w_for_L = np.random.randn(m // 2) * np.sqrt(2 / m)

# W = np.zeros((d, m))
# W[0 : d // 2, 0 : m // 2] = w_for_1
# W[d // 2 :, m // 2 :] = w_for_1
# # Weight["W1"] = torch.tensor(W, requires_grad=True)
# W1 = torch.tensor(W, requires_grad=True)

# W = np.zeros((m, 1))
# W[0 : m // 2, 0] = w_for_L
# W[m // 2 :, 0] = -w_for_L
# # Weight["W2"] = torch.tensor(W, requires_grad=True)
# W2 = torch.tensor(W, requires_grad=True)

# batchsize = 4
# x = np.random.normal(loc=0.0, scale=1.0, size=(batchsize, d // 2))
# x = np.concatenate([x, x], axis=1)
# A = np.random.uniform(low=0.0, high=1.0, size=(d, d))
# A = A + A.T
# y = np.sum(x @ A, axis=1)

# activation = torch.nn.ReLU()
# criterion = torch.nn.MSELoss()
# # optimizer = torch.optim.Adam([Weight["W1"], Weight["W2"]], lr=0.01)
# optimizer = torch.optim.Adam([W1, W2], lr=0.01)

# # output = activation(torch.matmul(torch.from_numpy(x), W1))
# # output = activation(torch.matmul(output, W2))
# output = activation(torch.matmul(torch.from_numpy(x), W1))
# output = torch.matmul(output, W2)
# loss = criterion(output[:, 0], torch.from_numpy(y))
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()


#%% unit test 4 -- test customized network
# d = 4
# L = 2
# m = 4
# mynn = NeuralNetwork(d=d, L=L, m=m)
# for para in mynn.parameters():
#     print(para)
# batchsize = 4
# x = np.zeros((batchsize, d))
# x = torch.from_numpy(x)
# y = mynn.forward(x)
# grad = mynn.GetGrad(x[0, :])
# print(grad)

#%% unit test 1
# from GameSetting import *

# ## Set the parameter of the game
# np.random.seed(12345)
# K = 4  # Total number of actions,
# T = 10  # Total number of periods
# d = 6  # the dimension of context
# A = np.random.normal(loc=0, scale=1, size=(d, d))

# bestagent = BestAgent(K, T, d, A)
# uniformagent = UniformAgent(K, T, d)

# for tt in range(1, T + 1):

#     # observe \{x_{t,a}\}_{a=1}^{k=1}
#     context_list = SampleContext(d, K)
#     realized_reward = GetRealReward(context_list, A)

#     # bestagent
#     best_ind = bestagent.Action(context_list)  # make a decision
#     best_reward = realized_reward[best_ind]  # play best_ind-th arm and observe reward
#     bestagent.Update(best_reward)

#     # uniformagent
#     uniform_ind = uniformagent.Action(context_list)  # make a decision
#     uniform_reward = realized_reward[uniform_ind]  # play uniform_ind-th arm and observe reward
#     uniformagent.Update(uniform_reward)

#     print(f"round index {tt}; best choose {best_ind}, reward is {best_reward}; uniform choose {uniform_ind}, reward is {uniform_reward}")

#%% unit test 2
# from GameSetting import *

# ## Set the parameter of the game
# np.random.seed(12345)
# K = 4  # Total number of actions,
# T = 5000  # Total number of periods
# d = 6  # the dimension of context
# A = np.random.normal(loc=0, scale=1, size=(d, d))

# neuralagent = NeuralAgent(K=K, T=T, d=d)
# bestagent = BestAgent(K, T, d, A)
# uniformagent = UniformAgent(K, T, d)
# for tt in range(1, T + 1):
#     # observe \{x_{t,a}\}_{a=1}^{k=1}
#     context_list = SampleContext(d, K)
#     realized_reward = GetRealReward(context_list, A)

#     # bestagent
#     best_ind = bestagent.Action(context_list)  # make a decision
#     best_reward = realized_reward[best_ind]  # play best_ind-th arm and observe reward
#     bestagent.Update(best_reward)

#     # uniformagent
#     uniform_ind = uniformagent.Action(context_list)  # make a decision
#     uniform_reward = realized_reward[uniform_ind]  # play uniform_ind-th arm and observe reward
#     uniformagent.Update(uniform_reward)

#     # neural agent
#     neural_ind = neuralagent.Action(context_list)
#     neural_reward = realized_reward[neural_ind]
#     neuralagent.Update(neural_reward)

#     print(f"round index {tt}; best choose {best_ind}, reward is {best_reward}; neural choose {neural_ind}, reward is {neural_reward}")
