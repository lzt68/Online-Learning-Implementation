import numpy as np
import random
from copy import deepcopy
from GameSetting import *
from NeuralNetworkRelatedFunction import *
import torch


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
        self.W0 = dict()
        for key in self.W.keys():
            self.W0[key] = deepcopy(self.W[key])
            self.W0[key].requires_grad_(requires_grad=False)

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


class NeuralAgent:
    def __init__(
        self,
        K: int,
        T: int,
        d: int,
        L: int = 2,
        m: int = 20,
        nu: float = 0.1,
        lambda_: float = 1,
        eta: float = 0.01,
        frequency: int = 50,
        batchsize: int = None,
        random_seed: int = 12345,
    ):
        """_summary_

        Args:
            K (int): Total number of actions
            T (int): Total number of periods
            d (int): The dimension of context
            L (int, optional): The number of hidden layer. Defaults to 2.
            m (int, optional): The number of neurals in each layer. Defaults to 20.
            nu (float, optional): Scale coefficient of variance. Defaults to 0.1.
            lambda_ (float, optional): Regularization of regression problem. Defaults to 1.
            eta (float, optional): Step size of the SGD. Defaults to 0.01.
            frequency (int, optional): The interval between two training epoches. Defaults to 50.
            batchsize (int, optional): The size of sample batch in SGD. Defaults to None.
            random_seed (int, optional) : The random seed of. Defaults to 12345
        """
        # random seed
        self.random_generator = random.Random()
        self.random_generator.seed(random_seed)
        self.random_seed = random_seed

        # the setup of the problem
        self.K = K
        self.T = T

        # the setup of neural network
        self.L = L
        self.m = m
        self.d = d
        self.p = m + m * d + m * m * (L - 2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mynn = NeuralNetwork(d=d, L=L, m=m, device=self.device)
        self.optimizer = torch.optim.SGD(self.mynn.parameters(), lr=eta)

        # the setup of the trainning process
        self.nu = nu
        self.lambda_ = self.lambda_
        self.eta = eta
        self.frequency = frequency
        self.batchsize = batchsize
        self.U = lambda_ * np.eye(self.p)

        # the history
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.predicted_reward = np.zeros(T)
        self.history_context = np.zeros((T, d))

    def Action(self, context_list: np.array) -> int:
        """Given the observed context of each arm, return the predicted arm

        Args:
            context_list (np.array): The observed context of each arm. context_list.shape = (K, d)

        Returns:
            int: the index of predicted arm, take value from 0, 1, ..., K-1
        """
        sample_reward_ = np.zeros(self.K)
        U_inverse = np.linalg.inv(self.U)
        for arm in range(1, self.K + 1):
            grad_arm = self.mynn.GetGrad(torch.from_numpy(context_list[arm - 1, :]))
            sigma_t_k2 = self.lambda_ * grad_arm.dot(U_inverse).dot(grad_arm)
            predict_reward = self.mynn.forward(torch.from_numpy(context_list[arm - 1, :]))
            sample_reward_[arm - 1] = self.random_generator.normalvariate(mu=predict_reward, sigma=np.sqrt(self.nu**2 * sigma_t_k2))
            # sample_reward_[arm - 1] = np.random.normal(loc=predict_reward, scale=np.sqrt(self.nu**2 * sigma_t_k2))
        ind = np.argmax(sample_reward_)

        # save the history
        self.history_action[self.t] = ind
        self.history_context[self.t, :] = context_list[ind, :]
        self.predicted_reward[self.t] = predict_reward[ind]

        return ind

    def Update(self, reward):
        self.history_reward[self.t] = reward
        ind = self.history_action[self.t]
        context = self.history_context[self.t, :]

        # update U
        grad_parameter = self.mynn.GetGrad(torch.from_numpy(context))
        self.U = self.U + grad_parameter.dot(grad_parameter.transpose()) / self.m

        # train the network
        if (self.t + 1) % self.frequency == 0:
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
