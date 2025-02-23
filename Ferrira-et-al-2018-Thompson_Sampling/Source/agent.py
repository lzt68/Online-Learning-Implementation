import sys

sys.path.append("./Source/")

import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd
from copy import deepcopy
from typing import Callable
from typing import Union

# if you have scip, you can use scip to solve the linear programme
# import pyscipopt
# from pyscipopt import quicksum

from scipy.optimize import linprog
from Metropolitan_Hastings import MHSampling


class ThompsonAgent_Fixed_Beta:
    def __init__(self, K: int, N: int, M: int, T: int, P_list: np.ndarray, A: np.ndarray, I_0: np.ndarray, random_seed: int = 12345):
        """Implement the 1st algorithm in Ferrira. We assume the prior distribution is uniform(0, 1),
        then we can derive the exact formula of posterior distribution,
        which is beta, to sample theta and further generate demand

        Args:
            K (int): Number of arms
            N (int): Number of products
            M (int): Number of resources
            T (int): Total rounds
            P_list (np.ndarray): The price of each product, whose shape is (K, N)
            A (np.ndarray): The unit cost of each product, whose shape is (N, M)
            I_0 (np.ndarray): Initial resource, whose shape is (M,)
            random_seed (int, optional): random seed. Defaults to 12345
        """
        assert P_list.shape[0] == K and P_list.shape[1] == N, "The shape of price doesn't match"
        assert A.shape[0] == N and A.shape[1] == M, "The shape of unit cost doesn't match"
        assert I_0.shape[0] == M, "The shape of initial resource doesn't match"

        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.P_list = P_list
        self.A = A
        self.I_0 = I_0
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        # initialize history
        self.H_P = np.zeros(shape=T)  # the index of pricing vector we used in each period
        self.H_D = np.zeros(shape=(T, N))  # the demand of products in each period
        self.H_I = np.zeros(shape=(T + 1, M))  # avaliable remained inventory in each period
        self.H_I[0, :] = np.float64(I_0)
        self.H_reward = np.zeros(T)  # the reward in each period
        self.H_bestX = np.zeros(shape=(T, K + 1))  # the best solution in each optimization

        # each realization of price vector, index of period,
        # corresponds to a estimate of theta
        self.H_alpha = np.zeros(shape=(T + 1, K, N))
        self.H_beta = np.zeros(shape=(T + 1, K, N))
        self.H_alpha[0, :, :] = 1 * np.ones(shape=(K, N))
        self.H_beta[0, :, :] = 1 * np.ones(shape=(K, N))

        # estimated theta in each round
        self.H_theta = np.zeros(shape=(T, K, N))

        # initialize the index of period
        self.t = 1

        # initialize the average consumption of resource
        self.c = I_0 / T

    def action(self):
        if not all(self.H_I[self.t - 1] >= 0):
            # resource are not enough
            price_offered_index = self.K + 1
            self.H_P[self.t - 1] = price_offered_index  # record the index of offered price
            return price_offered_index

        # first step, sample from posterior distribution
        # H_alpha[t-1, :, :], H_beta[t-1, :, :] is the history data from 0 to t
        # H_theta[t-1, :, :] is the sample theta we used in round t
        self.H_theta[self.t - 1, :, :] = self.sample_theta()

        # second step, solve the linear optimization problem
        bestx = self.get_optimal_solution(demand=self.H_theta[self.t - 1, :, :])

        # third step, offer price
        price_offered_index = self.random_generator.choice(np.arange(1, self.K + 2), p=bestx)

        # fourth step, update estimate of parameter
        self.H_P[self.t - 1] = price_offered_index  # record the index of offered price

        # # fourth step, record the constraint value in optimization
        # self.H_constraint_value[self.t - 1, 0] = np.sum(bestx[0 : self.K])
        # for jj in range(1, self.M + 1):
        #     con_coefficient = np.array(list(model.getValsLinear(constraint_index[jj]).values()))
        #     self.H_constraint_value[self.t - 1, jj] = np.sum(bestx[0 : self.K] * con_coefficient)

        # fourth step, record the optimal solution in this round
        self.H_bestX[self.t - 1, :] = bestx

        return price_offered_index

    def update(self, demand):
        # record the realized demand
        self.H_D[self.t - 1, :] = demand

        # record the reward
        price_offered_index = np.int64(self.H_P[self.t - 1])  # record the index of offered price
        if price_offered_index < self.K + 1:
            self.H_reward[self.t - 1] = self.P_list[price_offered_index - 1, :].dot(self.H_D[self.t - 1, :])
        else:  # the demand must be zero
            self.H_reward[self.t - 1] = 0

        # update the remaining inventory
        self.H_I[self.t] = self.H_I[self.t - 1] - np.transpose(self.A).dot(self.H_D[self.t - 1, :])

        # update the estimation of alpha and beta
        if price_offered_index < self.K + 1:
            # if demand = 1, then alpha plus 1; if demand = 0, then alpha remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_alpha[self.t, price_offered_index - 1, :] = self.H_alpha[self.t, price_offered_index - 1, :] + self.H_D[self.t - 1, :]

            # if demand = 1, then beta remained unchanged; if demand = 0, then beta plus 1
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]
            self.H_beta[self.t, price_offered_index - 1, :] = self.H_beta[self.t - 1, price_offered_index - 1, :] + np.ones(self.N) - self.H_D[self.t - 1, :]
        else:  # the demand must be zero, then all the estimate remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]

        # update the index of period
        self.t = self.t + 1

    def sample_theta(self):
        # use the history to sample theta from the posterior distribution

        # vectorize beta sample function to accelerate
        mybeta_generator = np.vectorize(self.random_generator.beta)
        theta = mybeta_generator(self.H_alpha[self.t - 1, :, :], self.H_beta[self.t - 1, :, :])
        return theta

    # use scip to solve the linear programme
    # def get_optimal_solution_(self, demand):
    #     # second step, optimize a linear function
    #     model = pyscipopt.Model("Optimization in Round {:d}".format(self.t))
    #     # generate decision variable
    #     x = {}
    #     for xindex in range(1, self.K + 1):
    #         x[xindex] = model.addVar(vtype="C", lb=0, ub=1, name="x{:d}".format(xindex))

    #     # second step, generate object function
    #     obj_coefficient = np.sum(demand * self.P_list, axis=1)  # obj_coefficient[k] = $\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1}$
    #     model.setObjective(quicksum(x[xindex] * obj_coefficient[xindex - 1] for xindex in range(1, self.K + 1)), "maximize")
    #     # objective = $\sum_{k=1}^K(\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1})x_{k}$

    #     # second step, add constraint x_1+...+x_k<=1
    #     constraint_index = {}
    #     constraint_index[0] = model.addCons(quicksum(x[xindex] for xindex in range(1, self.K + 1)) <= 1)

    #     # second step, for each resources, we require \sum_{k=1}^K\sum_{i=1}^N d_{i,k}a_{i,j}x_l<=c_j
    #     for jj in range(1, self.M + 1):
    #         con_coefficient = self.A[:, jj - 1].dot(np.transpose(demand))  # con_coefficient[k] = $\sum_{i=1}^N a_{i,j}d_{i,k+1}$
    #         constraint_index[jj] = model.addCons(quicksum(x[xindex] * con_coefficient[xindex - 1] for xindex in range(1, self.K + 1)) <= self.c[jj - 1])

    #     # second step, optimize the problem
    #     model.optimize()
    #     bestx = np.zeros(self.K + 1)  # p_{K+1} would force the demand be zero
    #     for xindex in range(1, self.K + 1):
    #         bestx[xindex - 1] = model.getVal(x[xindex])
    #     bestx[self.K] = 1 - np.sum(bestx[0 : self.K])
    #     eliminate_error = lambda x: 0 if np.abs(x) < 1e-10 else x  # there would be numerical error in the best solution
    #     bestx = np.array([eliminate_error(x) for x in bestx])
    #     bestx = bestx / np.sum(bestx)

    #     return bestx

    def get_optimal_solution(self, demand: np.ndarray):
        """call scipy.optimize.linprog to solve the linear programming problem
        min c@x
        s.t. Aub @ x <= bub
             Aeq @ x  = beq
             l <= x <= u
        In our case, we aim to maximize the problem, which is
        max \sum_{k=1}^K(\sum_{i=1}^N p_{ik}d_{ik}(t))x_k
        s.t. \sum_{k=1}^K(\sum_{i=1}^N a_{ij}d_{ik})x_k <= c_j, 1\le j \le M
             \sum_{k=1}^K x_k \le 1
             x_k \ge 0

        Args:
            demand (np.ndarray): Sampled value from the posterior distribution, whose shape is (K, N)
        """

        # generate coefficient of object function
        c = -np.sum(self.P_list * demand, axis=1)

        # generate coefficient matrix of the constraints
        Aub = np.zeros((self.M + 1, self.K))
        Aub[: self.M, :] = (demand @ self.A).T  # demand @ self.A is a K*M matrix, each row represents the consumption of each resource
        Aub[self.M, :] = np.ones(self.K)
        bub = np.zeros(self.M + 1)
        bub[: self.M] = self.c
        bub[self.M] = 1

        # solve the problem
        result = linprog(c, A_ub=Aub, b_ub=bub)

        # unzip the result
        bestx = np.zeros(self.K + 1)
        bestx[0 : self.K] = result.x
        bestx[self.K] = 1 - sum(bestx)
        eliminate_error = lambda x: 0 if x < 0 else x  # there would be numerical error in the best solution
        bestx = np.array([eliminate_error(x) for x in bestx])
        bestx = bestx / np.sum(bestx)

        return bestx


class ThompsonAgent_Fixed_MH:
    def __init__(self, K: int, N: int, M: int, T: int, P_list: np.ndarray, A: np.ndarray, I_0: np.ndarray, prior_list: Union[list, str] = "Default", MH_N: int = 5000, random_seed: int = 12345):
        """Implement the 1st algorithm in Ferrira. We assume the prior distribution of the theta is unifrom(0, 1).
        Though we can derive the explicit formula of the distribution,
        we still use the Metropolitan-Hastings algorithm to sample theta

        Args:
            K (int): Number of arms
            N (int): Number of products
            M (int): Number of resources
            T (int): Total rounds
            P_list (np.ndarray): The price of each product, whose shape is (K, N)
            A (np.ndarray): The unit cost of each product, whose shape is (N, M)
            I_0 (np.ndarray): Initial resource, whose shape is (M,)
            prior_list (Union[list, str], optional): The list containing the prior distribution of the theta, the length of list should be K.
            Defaults to "Default", which is unifrom distribution for all the arms
            MH_N (int, optional): The length of warm up phase in MH sampling. Defaults to 100.
            random_seed (int, optional): random seed. Defaults to 12345
        """
        assert P_list.shape[0] == K and P_list.shape[1] == N, "The shape of price doesn't match"
        assert A.shape[0] == N and A.shape[1] == M, "The shape of unit cost doesn't match"
        assert I_0.shape[0] == M, "The shape of initial resource doesn't match"

        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.P_list = P_list
        self.A = A
        self.I_0 = I_0
        self.MH_N = MH_N
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))
        if type(prior_list) == list:
            assert len(prior_list) == K, "The number of the prior distributions doesn't match with arm number"
            self.prior_list = prior_list
        elif prior_list == "Default":
            self.prior_list = [lambda x: 1.0 if np.all(x <= 1.0) and np.all(x >= 0.0) else 0.0] * K

        # initialize history
        self.H_P = np.zeros(shape=T)  # the index of pricing vector we used in each period
        self.H_D = np.zeros(shape=(T, N))  # the demand of products in each period
        self.H_I = np.zeros(shape=(T + 1, M))  # avaliable remained inventory in each period
        self.H_I[0, :] = np.float64(I_0)
        self.H_reward = np.zeros(T)  # the reward in each period
        self.H_bestX = np.zeros(shape=(T, K + 1))  # the best solution in each optimization

        # each realization of price vector, index of period,
        # corresponds to a estimate of theta
        self.H_alpha = np.zeros(shape=(T + 1, K, N))  # the times that the arm get pulled and consumption is 1
        self.H_beta = np.zeros(shape=(T + 1, K, N))  # the times that the arm get pulled and consumption is 0
        self.H_alpha[0, :, :] = 1 * np.ones(shape=(K, N))
        self.H_beta[0, :, :] = 1 * np.ones(shape=(K, N))

        # estimated theta in each round
        self.H_theta = np.zeros(shape=(T, K, N))

        # initialize the index of period
        self.t = 1

        # initialize the average consumption of resource
        self.c = I_0 / T

    def action(self):
        if not all(self.H_I[self.t - 1] >= 0):
            # resource are not enough
            price_offered_index = self.K + 1
            self.H_P[self.t - 1] = price_offered_index  # record the index of offered price
            return price_offered_index

        # first step, sample from posterior distribution
        # H_alpha[t-1, :, :], H_beta[t-1, :, :] is the history data from 0 to t
        # H_theta[t-1, :, :] is the sample theta we used in round t
        self.H_theta[self.t - 1, :, :] = self.sample_theta()
        self.H_theta[self.t - 1, :, :] = np.maximum(self.H_theta[self.t - 1, :, :], np.zeros((self.K, self.N)))
        self.H_theta[self.t - 1, :, :] = np.minimum(self.H_theta[self.t - 1, :, :], np.ones((self.K, self.N)))

        # second step, solve the linear optimization problem
        bestx = self.get_optimal_solution(demand=self.H_theta[self.t - 1, :, :])

        # third step, offer price
        price_offered_index = self.random_generator.choice(np.arange(1, self.K + 2), p=bestx)

        # fourth step, update estimate of parameter
        self.H_P[self.t - 1] = price_offered_index  # record the index of offered price

        # # fourth step, record the constraint value in optimization
        # self.H_constraint_value[self.t - 1, 0] = np.sum(bestx[0 : self.K])
        # for jj in range(1, self.M + 1):
        #     con_coefficient = np.array(list(model.getValsLinear(constraint_index[jj]).values()))
        #     self.H_constraint_value[self.t - 1, jj] = np.sum(bestx[0 : self.K] * con_coefficient)

        # fourth step, record the optimal solution in this round
        self.H_bestX[self.t - 1, :] = bestx

        return price_offered_index

    def update(self, demand):
        # record the realized demand
        self.H_D[self.t - 1, :] = demand

        # record the reward
        price_offered_index = np.int64(self.H_P[self.t - 1])  # record the index of offered price
        if price_offered_index < self.K + 1:
            self.H_reward[self.t - 1] = self.P_list[price_offered_index - 1, :].dot(self.H_D[self.t - 1, :])
        else:  # the demand must be zero
            self.H_reward[self.t - 1] = 0

        # update the remaining inventory
        self.H_I[self.t] = self.H_I[self.t - 1] - np.transpose(self.A).dot(self.H_D[self.t - 1, :])

        # update the estimation of alpha and beta
        if price_offered_index < self.K + 1:
            # if demand = 1, then alpha plus 1; if demand = 0, then alpha remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_alpha[self.t, price_offered_index - 1, :] = self.H_alpha[self.t, price_offered_index - 1, :] + self.H_D[self.t - 1, :]

            # if demand = 1, then beta remained unchanged; if demand = 0, then beta plus 1
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]
            self.H_beta[self.t, price_offered_index - 1, :] = self.H_beta[self.t - 1, price_offered_index - 1, :] + np.ones(self.N) - self.H_D[self.t - 1, :]
        else:  # the demand must be zero, then all the estimate remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]

        # update the index of period
        self.t = self.t + 1

    def sample_theta(self) -> np.ndarray:
        """sample theta from the posterior distribution

        Returns:
            np.ndarray: sampled value of theta whose shape is (K, N)
        """
        # use the history to sample theta from the posterior distribution
        theta = np.zeros((self.K, self.N))
        for kk in range(self.K):
            # calculate the
            def g(x):
                density = self.prior_list[kk](x)
                density *= np.prod(x ** self.H_alpha[self.t - 1, kk, :]) * np.prod((1.0 - x) ** self.H_beta[self.t - 1, kk, :])
                return density

            # make each sampling adopt   different random seed
            theta[kk, :] = MHSampling(N=self.MH_N, M=1, d=self.N, g=g, verbose=False, random_seed=kk * self.T + self.t)[0, :]
        return theta

    # use scip to solve the linear programme
    # def get_optimal_solution_(self, demand):
    #     model = pyscipopt.Model("Optimization in Round {:d}".format(self.t))
    #     # generate decision variable
    #     x = {}
    #     for xindex in range(1, self.K + 1):
    #         x[xindex] = model.addVar(vtype="C", lb=0, ub=1, name="x{:d}".format(xindex))

    #     # generate object function
    #     obj_coefficient = np.sum(demand * self.P_list, axis=1)  # obj_coefficient[k] = $\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1}$
    #     model.setObjective(quicksum(x[xindex] * obj_coefficient[xindex - 1] for xindex in range(1, self.K + 1)), "maximize")
    #     # objective = $\sum_{k=1}^K(\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1})x_{k}$

    #     # add constraint x_1+...+x_k<=1
    #     constraint_index = {}
    #     constraint_index[0] = model.addCons(quicksum(x[xindex] for xindex in range(1, self.K + 1)) <= 1)

    #     # add constraint, for each resources, we require \sum_{k=1}^K\sum_{i=1}^N d_{i,k}a_{i,j}x_l<=c_j
    #     for jj in range(1, self.M + 1):
    #         con_coefficient = self.A[:, jj - 1].dot(np.transpose(demand))  # con_coefficient[k] = $\sum_{i=1}^N a_{i,j}d_{i,k+1}$
    #         constraint_index[jj] = model.addCons(quicksum(x[xindex] * con_coefficient[xindex - 1] for xindex in range(1, self.K + 1)) <= self.c[jj - 1])

    #     # optimize the problem
    #     model.optimize()
    #     bestx = np.zeros(self.K + 1)  # p_{K+1} would force the demand be zero
    #     for xindex in range(1, self.K + 1):
    #         bestx[xindex - 1] = model.getVal(x[xindex])
    #     bestx[self.K] = 1 - np.sum(bestx[0 : self.K])
    #     eliminate_error = lambda x: 0 if np.abs(x) < 1e-10 else x  # there would be numerical error in the best solution
    #     bestx = np.array([eliminate_error(x) for x in bestx])
    #     bestx = bestx / np.sum(bestx)

    #     return bestx

    def get_optimal_solution(self, demand: np.ndarray):
        """call scipy.optimize.linprog to solve the linear programming problem
        min c@x
        s.t. Aub @ x <= bub
             Aeq @ x  = beq
             l <= x <= u
        In our case, we aim to maximize the problem, which is
        max \sum_{k=1}^K(\sum_{i=1}^N p_{ik}d_{ik}(t))x_k
        s.t. \sum_{k=1}^K(\sum_{i=1}^N a_{ij}d_{ik})x_k <= c_j, 1\le j \le M
             \sum_{k=1}^K x_k \le 1
             x_k \ge 0

        Args:
            demand (np.ndarray): Sampled value from the posterior distribution, whose shape is (K, N)
        """

        # generate coefficient of object function
        c = -np.sum(self.P_list * demand, axis=1)

        # generate coefficient matrix of the constraints
        Aub = np.zeros((self.M + 1, self.K))
        Aub[: self.M, :] = (demand @ self.A).T  # demand @ self.A is a K*M matrix, each row represents the consumption of each resource
        Aub[self.M, :] = np.ones(self.K)
        bub = np.zeros(self.M + 1)
        bub[: self.M] = self.c
        bub[self.M] = 1

        # solve the problem
        result = linprog(c=c, A_ub=Aub, b_ub=bub)

        # unzip the result
        bestx = np.zeros(self.K + 1)
        bestx[0 : self.K] = result.x
        bestx[self.K] = 1 - sum(bestx)
        eliminate_error = lambda x: 0 if x < 0 else x  # there would be numerical error in the best solution
        bestx = np.array([eliminate_error(x) for x in bestx])
        bestx = bestx / np.sum(bestx)

        return bestx


class ThompsonAgent_Update_Beta:
    def __init__(self, K: int, N: int, M: int, T: int, P_list: np.ndarray, A: np.ndarray, I_0: np.ndarray, random_seed: int = 12345):
        """Implement the 1st algorithm in Ferrira. We assume the prior distribution is uniform(0, 1),
        then we can derive the exact formula of posterior distribution,
        which is beta, to sample theta and further generate demand

        Args:
            K (int): Number of arms
            N (int): Number of products
            M (int): Number of resources
            T (int): Total rounds
            P_list (np.ndarray): The price of each product, whose shape is (K, N)
            A (np.ndarray): The unit cost of each product, whose shape is (N, M)
            I_0 (np.ndarray): Initial resource, whose shape is (M,)
            random_seed (int, optional): random seed. Defaults to 12345
        """
        assert P_list.shape[0] == K and P_list.shape[1] == N, "The shape of price doesn't match"
        assert A.shape[0] == N and A.shape[1] == M, "The shape of unit cost doesn't match"
        assert I_0.shape[0] == M, "The shape of initial resource doesn't match"

        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.P_list = P_list
        self.A = A
        self.I_0 = I_0
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        # initialize history
        self.H_P = np.zeros(shape=T)  # the index of pricing vector we used in each period
        self.H_D = np.zeros(shape=(T, N))  # the demand of products in each period
        self.H_I = np.zeros(shape=(T + 1, M))  # avaliable remaining inventory in each period
        self.H_I[0, :] = np.float64(I_0)
        self.H_reward = np.zeros(T)  # the reward in each period
        self.H_bestX = np.zeros(shape=(T, K + 1))  # the best solution in each optimization

        # each realization of price vector, index of period,
        # corresponds to a estimate of theta
        self.H_alpha = np.zeros(shape=(T + 1, K, N))
        self.H_beta = np.zeros(shape=(T + 1, K, N))
        self.H_alpha[0, :, :] = 1 * np.ones(shape=(K, N))
        self.H_beta[0, :, :] = 1 * np.ones(shape=(K, N))

        # initialize the constraint value in each round
        # M kinds of resources correspond to M constraints, and one more constraint is x1 + ... + xN <=1
        self.H_constraint_value = np.zeros(shape=(T, M + 1))

        # estimated theta in each round
        self.H_theta = np.zeros(shape=(T, K, N))

        # initialize the index of period
        self.t = 1

        # initialize the average consumption of resource
        self.c = I_0 / T

    def action(self):
        if not all(self.H_I[self.t - 1] >= 0):
            # resource are not enough
            price_offered_index = self.K + 1
            self.H_P[self.t - 1] = price_offered_index  # record the index of offered price
            return price_offered_index

        # first step, sample from posterior distribution
        # H_alpha[t-1, :, :], H_beta[t-1, :, :] is the history data from 0 to t
        # H_theta[t-1, :, :] is the sample theta we used in round t
        self.H_theta[self.t - 1, :, :] = self.sample_theta()

        # second step, solve the linear optimization problem
        bestx = self.get_optimal_solution(demand=self.H_theta[self.t - 1, :, :])

        # third step, offer price
        price_offered_index = self.random_generator.choice(np.arange(1, self.K + 2), p=bestx)

        # fourth step, update estimate of parameter
        self.H_P[self.t - 1] = price_offered_index  # record the index of offered price

        # # fourth step, record the constraint value in optimization
        # self.H_constraint_value[self.t - 1, 0] = np.sum(bestx[0 : self.K])
        # for jj in range(1, self.M + 1):
        #     con_coefficient = np.array(list(model.getValsLinear(constraint_index[jj]).values()))
        #     self.H_constraint_value[self.t - 1, jj] = np.sum(bestx[0 : self.K] * con_coefficient)

        # fourth step, record the optimal solution in this round
        self.H_bestX[self.t - 1, :] = bestx

        return price_offered_index

    def update(self, demand):
        # record the realized demand
        self.H_D[self.t - 1, :] = demand

        # record the reward
        price_offered_index = np.int64(self.H_P[self.t - 1])  # record the index of offered price
        if price_offered_index < self.K + 1:
            self.H_reward[self.t - 1] = self.P_list[price_offered_index - 1, :].dot(self.H_D[self.t - 1, :])
        else:  # the demand must be zero
            self.H_reward[self.t - 1] = 0

        # update the remaining inventory
        self.H_I[self.t] = self.H_I[self.t - 1] - np.transpose(self.A).dot(self.H_D[self.t - 1, :])

        # update the estimation of alpha and beta
        if price_offered_index < self.K + 1:
            # if demand = 1, then alpha plus 1; if demand = 0, then alpha remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_alpha[self.t, price_offered_index - 1, :] = self.H_alpha[self.t, price_offered_index - 1, :] + self.H_D[self.t - 1, :]

            # if demand = 1, then beta remained unchanged; if demand = 0, then beta plus 1
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]
            self.H_beta[self.t, price_offered_index - 1, :] = self.H_beta[self.t - 1, price_offered_index - 1, :] + np.ones(self.N) - self.H_D[self.t - 1, :]
        else:  # the demand must be zero, then all the estimate remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]

        # update the index of period
        self.t = self.t + 1

    def sample_theta(self):
        # use the history to sample theta from the posterior distribution

        # vectorize beta sample function to accelerate
        mybeta_generator = np.vectorize(self.random_generator.beta)
        theta = mybeta_generator(self.H_alpha[self.t - 1, :, :], self.H_beta[self.t - 1, :, :])
        return theta

    # use scip to solve the linear programme
    # def get_optimal_solution_(self, demand):
    #     # second step, optimize a linear function
    #     model = pyscipopt.Model("Optimization in Round {:d}".format(self.t))
    #     # generate decision variable
    #     x = {}
    #     for xindex in range(1, self.K + 1):
    #         x[xindex] = model.addVar(vtype="C", lb=0, ub=1, name="x{:d}".format(xindex))

    #     # second step, generate object function
    #     obj_coefficient = np.sum(demand * self.P_list, axis=1)  # obj_coefficient[k] = $\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1}$
    #     model.setObjective(quicksum(x[xindex] * obj_coefficient[xindex - 1] for xindex in range(1, self.K + 1)), "maximize")
    #     # objective = $\sum_{k=1}^K(\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1})x_{k}$

    #     # second step, add constraint x_1+...+x_k<=1
    #     constraint_index = {}
    #     constraint_index[0] = model.addCons(quicksum(x[xindex] for xindex in range(1, self.K + 1)) <= 1)

    #     # second step, for each resources, we require \sum_{k=1}^K\sum_{i=1}^N d_{i,k}a_{i,j}x_l<=c_j
    #     for jj in range(1, self.M + 1):
    #         con_coefficient = self.A[:, jj - 1].dot(np.transpose(demand))  # con_coefficient[k] = $\sum_{i=1}^N a_{i,j}d_{i,k+1}$
    #         constraint_index[jj] = model.addCons(quicksum(x[xindex] * con_coefficient[xindex - 1] for xindex in range(1, self.K + 1)) <= self.c[jj - 1])

    #     # second step, optimize the problem
    #     model.optimize()
    #     bestx = np.zeros(self.K + 1)  # p_{K+1} would force the demand be zero
    #     for xindex in range(1, self.K + 1):
    #         bestx[xindex - 1] = model.getVal(x[xindex])
    #     bestx[self.K] = 1 - np.sum(bestx[0 : self.K])
    #     eliminate_error = lambda x: 0 if np.abs(x) < 1e-10 else x  # there would be numerical error in the best solution
    #     bestx = np.array([eliminate_error(x) for x in bestx])
    #     bestx = bestx / np.sum(bestx)

    #     return bestx

    def get_optimal_solution(self, demand: np.ndarray):
        """call scipy.optimize.linprog to solve the linear programming problem
        min c@x
        s.t. Aub @ x <= bub
             Aeq @ x  = beq
             l <= x <= u
        In our case, we aim to maximize the problem, which is
        max \sum_{k=1}^K(\sum_{i=1}^N p_{ik}d_{ik}(t))x_k
        s.t. \sum_{k=1}^K(\sum_{i=1}^N a_{ij}d_{ik})x_k <= c_j, 1\le j \le M
             \sum_{k=1}^K x_k \le 1
             x_k \ge 0

        Args:
            demand (np.ndarray): Sampled value from the posterior distribution, whose shape is (K, N)
        """

        # generate coefficient of object function
        c = -np.sum(self.P_list * demand, axis=1)

        # generate coefficient matrix of the constraints
        Aub = np.zeros((self.M + 1, self.K))
        Aub[: self.M, :] = (demand @ self.A).T  # demand @ self.A is a K*M matrix, each row represents the consumption of each resource
        Aub[self.M, :] = np.ones(self.K)
        bub = np.zeros(self.M + 1)
        bub[: self.M] = self.H_I[self.t - 1] / (self.T - self.t + 1)
        bub[self.M] = 1

        # solve the problem
        result = linprog(c, A_ub=Aub, b_ub=bub)

        # unzip the result
        bestx = np.zeros(self.K + 1)
        bestx[0 : self.K] = result.x
        bestx[self.K] = 1 - sum(bestx)
        eliminate_error = lambda x: 0 if x < 0 else x  # there would be numerical error in the best solution
        bestx = np.array([eliminate_error(x) for x in bestx])
        bestx = bestx / np.sum(bestx)

        return bestx


class ThompsonAgent_Update_MH:
    def __init__(self, K: int, N: int, M: int, T: int, P_list: np.ndarray, A: np.ndarray, I_0: np.ndarray, prior_list: Union[list, str] = "Default", MH_N: int = 5000, random_seed: int = 12345):
        """Implement the 1st algorithm in Ferrira. We assume the prior distribution of the theta is unifrom(0, 1).
        Though we can derive the explicit formula of the distribution,
        we still use the Metropolitan-Hastings algorithm to sample theta

        Args:
            K (int): Number of arms
            N (int): Number of products
            M (int): Number of resources
            T (int): Total rounds
            P_list (np.ndarray): The price of each product, whose shape is (K, N)
            A (np.ndarray): The unit cost of each product, whose shape is (N, M)
            I_0 (np.ndarray): Initial resource, whose shape is (M,)
            prior_list (Union[list, str], optional): The list containing the prior distribution of the theta, the length of list should be K.
            Defaults to "Default", which is unifrom distribution for all the arms
            MH_N (int, optional): The length of warm up phase in MH sampling. Defaults to 100.
            random_seed (int, optional): random seed. Defaults to 12345
        """
        assert P_list.shape[0] == K and P_list.shape[1] == N, "The shape of price doesn't match"
        assert A.shape[0] == N and A.shape[1] == M, "The shape of unit cost doesn't match"
        assert I_0.shape[0] == M, "The shape of initial resource doesn't match"

        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.P_list = P_list
        self.A = A
        self.I_0 = I_0
        self.MH_N = MH_N
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))
        if type(prior_list) == list:
            assert len(prior_list) == K, "The number of the prior distributions doesn't match with arm number"
            self.prior_list = prior_list
        elif prior_list == "Default":
            self.prior_list = [lambda x: 1.0 if np.all(x <= 1.0) and np.all(x >= 0.0) else 0.0] * K

        # initialize history
        self.H_P = np.zeros(shape=T)  # the index of pricing vector we used in each period
        self.H_D = np.zeros(shape=(T, N))  # the demand of products in each period
        self.H_I = np.zeros(shape=(T + 1, M))  # avaliable remained inventory in each period
        self.H_I[0, :] = np.float64(I_0)
        self.H_reward = np.zeros(T)  # the reward in each period
        self.H_bestX = np.zeros(shape=(T, K + 1))  # the best solution in each optimization

        # each realization of price vector, index of period,
        # corresponds to a estimate of theta
        self.H_alpha = np.zeros(shape=(T + 1, K, N))  # the times that the arm get pulled and consumption is 1
        self.H_beta = np.zeros(shape=(T + 1, K, N))  # the times that the arm get pulled and consumption is 0
        self.H_alpha[0, :, :] = 1 * np.ones(shape=(K, N))
        self.H_beta[0, :, :] = 1 * np.ones(shape=(K, N))

        # initialize the constraint value in each round
        # M kinds of resources correspond to M constraints, and one more constraint is x1 + ... + xN <=1
        self.H_constraint_value = np.zeros(shape=(T, M + 1))

        # estimated theta in each round
        self.H_theta = np.zeros(shape=(T, K, N))

        # initialize the index of period
        self.t = 1

        # initialize the average consumption of resource
        self.c = I_0 / T

    def action(self):
        if not all(self.H_I[self.t - 1] >= 0):
            # resource are not enough
            price_offered_index = self.K + 1
            self.H_P[self.t - 1] = price_offered_index  # record the index of offered price
            return price_offered_index

        # first step, sample from posterior distribution
        # H_alpha[t-1, :, :], H_beta[t-1, :, :] is the history data from 0 to t
        # H_theta[t-1, :, :] is the sample theta we used in round t
        self.H_theta[self.t - 1, :, :] = self.sample_theta()
        self.H_theta[self.t - 1, :, :] = np.maximum(self.H_theta[self.t - 1, :, :], np.zeros((self.K, self.N)))
        self.H_theta[self.t - 1, :, :] = np.minimum(self.H_theta[self.t - 1, :, :], np.ones((self.K, self.N)))

        # second step, solve the linear optimization problem
        bestx = self.get_optimal_solution(demand=self.H_theta[self.t - 1, :, :])

        # third step, offer price
        price_offered_index = self.random_generator.choice(np.arange(1, self.K + 2), p=bestx)

        # fourth step, update estimate of parameter
        self.H_P[self.t - 1] = price_offered_index  # record the index of offered price

        # # fourth step, record the constraint value in optimization
        # self.H_constraint_value[self.t - 1, 0] = np.sum(bestx[0 : self.K])
        # for jj in range(1, self.M + 1):
        #     con_coefficient = np.array(list(model.getValsLinear(constraint_index[jj]).values()))
        #     self.H_constraint_value[self.t - 1, jj] = np.sum(bestx[0 : self.K] * con_coefficient)

        # fourth step, record the optimal solution in this round
        self.H_bestX[self.t - 1, :] = bestx

        return price_offered_index

    def update(self, demand):
        # record the realized demand
        self.H_D[self.t - 1, :] = demand

        # record the reward
        price_offered_index = np.int64(self.H_P[self.t - 1])  # record the index of offered price
        if price_offered_index < self.K + 1:
            self.H_reward[self.t - 1] = self.P_list[price_offered_index - 1, :].dot(self.H_D[self.t - 1, :])
        else:  # the demand must be zero
            self.H_reward[self.t - 1] = 0

        # update the remaining inventory
        self.H_I[self.t] = self.H_I[self.t - 1] - np.transpose(self.A).dot(self.H_D[self.t - 1, :])

        # update the estimation of alpha and beta
        if price_offered_index < self.K + 1:
            # if demand = 1, then alpha plus 1; if demand = 0, then alpha remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_alpha[self.t, price_offered_index - 1, :] = self.H_alpha[self.t, price_offered_index - 1, :] + self.H_D[self.t - 1, :]

            # if demand = 1, then beta remained unchanged; if demand = 0, then beta plus 1
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]
            self.H_beta[self.t, price_offered_index - 1, :] = self.H_beta[self.t - 1, price_offered_index - 1, :] + np.ones(self.N) - self.H_D[self.t - 1, :]
        else:  # the demand must be zero, then all the estimate remain unchanged
            self.H_alpha[self.t, :, :] = self.H_alpha[self.t - 1, :, :]
            self.H_beta[self.t, :, :] = self.H_beta[self.t - 1, :, :]

        # update the index of period
        self.t = self.t + 1

    def sample_theta(self) -> np.ndarray:
        """sample theta from the posterior distribution

        Returns:
            np.ndarray: sampled value of theta whose shape is (K, N)
        """
        # use the history to sample theta from the posterior distribution
        theta = np.zeros((self.K, self.N))
        for kk in range(self.K):
            # calculate the
            def g(x):
                density = self.prior_list[kk](x)
                density *= np.prod(x ** self.H_alpha[self.t - 1, kk, :]) * np.prod((1.0 - x) ** self.H_beta[self.t - 1, kk, :])
                return density

            # make each sampling adopt   different random seed
            theta[kk, :] = MHSampling(N=self.MH_N, M=1, d=self.N, g=g, verbose=False, random_seed=kk * self.T + self.t)[0, :]
        return theta

    # use scip to solve the linear programme
    # def get_optimal_solution_(self, demand):
    #     model = pyscipopt.Model("Optimization in Round {:d}".format(self.t))
    #     # generate decision variable
    #     x = {}
    #     for xindex in range(1, self.K + 1):
    #         x[xindex] = model.addVar(vtype="C", lb=0, ub=1, name="x{:d}".format(xindex))

    #     # generate object function
    #     obj_coefficient = np.sum(demand * self.P_list, axis=1)  # obj_coefficient[k] = $\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1}$
    #     model.setObjective(quicksum(x[xindex] * obj_coefficient[xindex - 1] for xindex in range(1, self.K + 1)), "maximize")
    #     # objective = $\sum_{k=1}^K(\sum_{i=1}^N d_{i,k+1}(t)p_{i,k+1})x_{k}$

    #     # add constraint x_1+...+x_k<=1
    #     constraint_index = {}
    #     constraint_index[0] = model.addCons(quicksum(x[xindex] for xindex in range(1, self.K + 1)) <= 1)

    #     # add constraint, for each resources, we require \sum_{k=1}^K\sum_{i=1}^N d_{i,k}a_{i,j}x_l<=c_j
    #     for jj in range(1, self.M + 1):
    #         con_coefficient = self.A[:, jj - 1].dot(np.transpose(demand))  # con_coefficient[k] = $\sum_{i=1}^N a_{i,j}d_{i,k+1}$
    #         constraint_index[jj] = model.addCons(quicksum(x[xindex] * con_coefficient[xindex - 1] for xindex in range(1, self.K + 1)) <= self.c[jj - 1])

    #     # optimize the problem
    #     model.optimize()
    #     bestx = np.zeros(self.K + 1)  # p_{K+1} would force the demand be zero
    #     for xindex in range(1, self.K + 1):
    #         bestx[xindex - 1] = model.getVal(x[xindex])
    #     bestx[self.K] = 1 - np.sum(bestx[0 : self.K])
    #     eliminate_error = lambda x: 0 if np.abs(x) < 1e-10 else x  # there would be numerical error in the best solution
    #     bestx = np.array([eliminate_error(x) for x in bestx])
    #     bestx = bestx / np.sum(bestx)

    #     return bestx

    def get_optimal_solution(self, demand: np.ndarray):
        """call scipy.optimize.linprog to solve the linear programming problem
        min c@x
        s.t. Aub @ x <= bub
             Aeq @ x  = beq
             l <= x <= u
        In our case, we aim to maximize the problem, which is
        max \sum_{k=1}^K(\sum_{i=1}^N p_{ik}d_{ik}(t))x_k
        s.t. \sum_{k=1}^K(\sum_{i=1}^N a_{ij}d_{ik})x_k <= c_j, 1\le j \le M
             \sum_{k=1}^K x_k \le 1
             x_k \ge 0

        Args:
            demand (np.ndarray): Sampled value from the posterior distribution, whose shape is (K, N)
        """

        # generate coefficient of object function
        c = -np.sum(self.P_list * demand, axis=1)

        # generate coefficient matrix of the constraints
        Aub = np.zeros((self.M + 1, self.K))
        Aub[: self.M, :] = (demand @ self.A).T  # demand @ self.A is a K*M matrix, each row represents the consumption of each resource
        Aub[self.M, :] = np.ones(self.K)
        bub = np.zeros(self.M + 1)
        bub[: self.M] = self.H_I[self.t - 1] / (self.T - self.t + 1)
        bub[self.M] = 1

        # solve the problem
        result = linprog(c=c, A_ub=Aub, b_ub=bub)

        # unzip the result
        bestx = np.zeros(self.K + 1)
        bestx[0 : self.K] = result.x
        bestx[self.K] = 1 - sum(bestx)
        eliminate_error = lambda x: 0 if x < 0 else x  # there would be numerical error in the best solution
        bestx = np.array([eliminate_error(x) for x in bestx])
        bestx = bestx / np.sum(bestx)

        return bestx


class UniformAgent:
    # this agent would uniformly pick pricing vector
    def __init__(self, K, N, M, T, P_list, A, I_0, random_seed: int = 12345):
        self.K = K
        self.N = N
        self.M = M
        self.T = T
        self.P_list = P_list
        self.A = A
        self.I_0 = I_0
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        # initialize history
        self.H_P = np.zeros(shape=T)  # the index of pricing vector we used in each period
        self.H_D = np.zeros(shape=(T, N))  # the demand of products in each period
        self.H_I = np.zeros(shape=(T + 1, M))  # avaliable remained inventory in each period
        self.H_I[0, :] = np.float64(I_0)
        self.H_reward = np.zeros(T)  # the reward in each period

        # initialize the index of period
        self.t = 1

    def action(self):
        if not all(self.H_I[self.t - 1] >= 0):
            # resource are not enough
            price_offered_index = self.K + 1
            self.H_P[self.t - 1] = price_offered_index  # record the index of offered price
            return price_offered_index

        price_offered_index = self.random_generator.choice(np.arange(1, self.K + 2))
        # the maximum value of np.random.randint(low, high) would be high - 1
        self.H_P[self.t - 1] = price_offered_index
        return price_offered_index

    def update(self, demand):
        # record the realized demand
        self.H_D[self.t - 1, :] = demand

        # record the reward
        price_offered_index = np.int64(self.H_P[self.t - 1])  # record the index of offered price
        if price_offered_index < self.K + 1:
            self.H_reward[self.t - 1] = self.P_list[price_offered_index - 1, :].dot(self.H_D[self.t - 1, :])
        else:  # the demand must be zero
            self.H_reward[self.t - 1] = 0

        # update the remaining inventory
        self.H_I[self.t] = self.H_I[self.t - 1] - np.transpose(self.A).dot(self.H_D[self.t - 1, :])

        # update the index of period
        self.t = self.t + 1


#%% unit test 1
# K = 2
# M = 2
# N = 2
# T = 10
# MH_N = 5000
# P_list = np.array([29.9, 34.9])
# A = np.ones((N, M))
# agent = ThompsonAgent_Fixed_MH(K=K, N=N, M=M, T=T, P_list=P_list, A=A, I_0=np.ones(M) * T * 2, MH_N=MH_N)
# agent.H_alpha[0, :, :] = np.array([[1, 2], [1, 1]]) * 100
# agent.H_beta[0, :, :] = np.array([[1, 1], [2, 2]]) * 100
# theta = agent.sample_theta()
# print(theta)

#%% unit test 2, check whether pyscip and scipy.optimize.linprog return the same solution
# # Generate parameters
# np.random.seed(12345)

# # the maximum value of np.random.randint(low, high) would be high - 1
# K = 4  # Total number of available price vectors,
# N = 3  # Total number of products
# M = 3  # Total number of resource
# T = 10  # Total number of periods
# P_list = np.float64(np.random.randint(low=1, high=10, size=(K, N)))
# c = np.random.randint(low=10, high=15, size=M)
# I_0 = c * T
# A = np.float64(np.random.randint(low=10, high=15, size=(N, M)))
# theta = np.random.beta(a=2, b=1, size=(K, N))

# agent = ThompsonAgent_Fixed_MH(K, N, M, T, P_list, A, I_0)
# bestx = agent.get_optimal_solution(demand=theta)
# print(f"scip.optimize.linprog, {bestx}")

# agent = ThompsonAgent_Fixed_MH(K, N, M, T, P_list, A, I_0)
# bestx = agent.get_optimal_solution_(demand=theta)
# print(f"pyscip, {bestx}")
