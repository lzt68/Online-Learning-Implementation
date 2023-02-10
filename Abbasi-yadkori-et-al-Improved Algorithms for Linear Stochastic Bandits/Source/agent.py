from typing import Union

import numpy as np
from numpy import linalg
from copy import deepcopy
from scipy.optimize import minimize


def get_factorization(matrix: np.ndarray):
    # calculate the unitarian factorization of a 2-d positive semi definite matrix
    # matrix = $U\Lambda U^T$, where \Lambda is diagnoal, \Lambda[0][0] > \Lambda[1][1]
    # U is unitary matrix
    assert matrix.shape[0] == 2 and matrix.shape[1] == 2, "not a 2-d matrix"
    assert np.abs(matrix[0][1] - matrix[1][0]) < 1e-8, "not a symmetric matrix"
    assert matrix[0][0] > 0 and matrix[1][1] > 0 and matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2 > 0, "not positive semidefinite"

    if np.abs(matrix[0][1]) < 1e-8:  # diagnoal matrix
        if matrix[0][0] > matrix[1][1]:
            lambda1 = matrix[0][0]
            lambda2 = matrix[1][1]
            U = np.eye(2)
            return lambda1, lambda2, U
        else:
            lambda1 = matrix[1][1]
            lambda2 = matrix[0][0]
            U = np.zeros((2, 2))
            U[0][1] = 1.0
            U[1][0] = 1.0
            return lambda1, lambda2, U
    else:
        a11 = matrix[0][0]
        a22 = matrix[1][1]
        a = matrix[0][1]
        lambda1 = ((a11 + a22) + np.sqrt((a11 - a22) ** 2 + 4 * a**2)) / 2
        lambda2 = ((a11 + a22) - np.sqrt((a11 - a22) ** 2 + 4 * a**2)) / 2

        U = np.zeros((2, 2))
        matrix_temp = matrix - np.eye(2) * lambda1
        U[0][0] = matrix_temp[0][1] / matrix_temp[0][0]
        U[1][0] = -1
        U[0:, 0] = U[0:, 0] / np.sqrt(U[0][0] ** 2 + U[1][0] ** 2)
        matrix_temp = matrix - np.eye(2) * lambda2
        U[0][1] = matrix_temp[0][1] / matrix_temp[0][0]
        U[1][1] = -1
        U[0:, 1] = U[0:, 1] / np.sqrt(U[0][1] ** 2 + U[1][1] ** 2)

        assert np.max(np.abs(U @ np.array([[lambda1, 0.0], [0.0, lambda2]]) @ U.T - matrix)) < 1e-6, "fail to factorize"

        return lambda1, lambda2, U


def get_xy(lambda1: float, lambda2: float, U: np.ndarray, x0: np.ndarray, beta: float, t: Union[float, np.ndarray]):
    """use the return value of function get_factorization to calculate the position of point
    Equation of ellipsoid (x-x_0)^TU[[lambda_1, 0], [0, lambda2]]U^T(x-x0)=beta

    Args:
        lambda1 (float): bigger eigen value of the matrix
        lambda2 (float): smaller eigen value of the matrix
        U (np.ndarray): the unitary matrix
        x0 (np.ndarray): the center of ellipsoid
        beta (np.ndarray): the number
        t (Union[float, np.ndarray]): the parameter of the point

    Returns:
        x(np.ndarray): each column of x represents a point in the 2-d space
    """
    x1 = np.expand_dims(np.sqrt(beta) / np.sqrt(lambda1) * np.sin(t), axis=0)
    x2 = np.expand_dims(np.sqrt(beta) / np.sqrt(lambda2) * np.cos(t), axis=0)
    x = np.concatenate([x1, x2], axis=0)

    if type(t) == float:
        x = U @ x + x0
    else:
        x = U @ x + x0[:, np.newaxis]
    return x


class ConfidenceBall_Agent(object):
    def __init__(self) -> None:
        pass


class OFUL_Agent(object):
    def __init__(self, R: float, lambda_: float, L: float, S: float, delta: float) -> None:
        """Initialization of OFUL algorithm,
        O: Optimism, U: Uncertainty, F: Face, L: Linear
        We assume the action space is always the unit ball in 2-d space

        Args:
            R (float): The nois is conditionally R-subgaussian random variable
            _lambda (float): Regularization coefficient
            L (float): $\|X_t\|_2 \le L$
            S (float): $\|\theta_*\|_2 \le S$
            delta (float): tolerance of failure probability
        """
        self.d = 2
        self.R = R
        self.lambda_ = lambda_
        self.L = L
        self.S = S
        self.delta = delta

        self.H_action = list()  # history of reward
        self.H_reward = list()  # history of action

        # $\hat{\theta}_t = A_t^{-1} B_t$
        self.V_t = lambda_ * np.eye(self.d)  # $A_t=\sum_{s=1}^t x_sx_s^T+\lambda I$
        self.Y_t = np.zeros(2)  # $B_t=\sum_{s=1}^t y_sx_s$

        self.t = 0

    def action(self):
        tilde_theta = self.get_theta()
        tilde_theta_l2 = linalg.norm(tilde_theta, ord=2)
        if np.abs(tilde_theta) < 1e-6:
            act = np.zeros(self.d)
        else:
            act = tilde_theta / tilde_theta_l2
        self.H_action.append(act)
        return act

    def update(self, reward):
        action = self.H_action[self.t]
        self.V_t = self.V_t + action[:, np.newaxis] @ action[np.newaxis, :]
        self.Y_t = self.Y_t + action * reward
        self.t += 1

    def get_theta(self):
        # solve the problem
        # \tilde{\theta}_t=\arg\max_{\theta\in C_{t-1}}\|\theta\|_2

        # calculate the $C_{t-1}$, including $V_{t-1}$, $\beta_{t-1}$, $\hat{\theta}$
        theta_hat_t = np.linalg.solve(self.V_t, self.Y_t)
        beta_t = self.R * np.sqrt(self.d * np.log((1 + t * self.L**2 / self.lambda_) / self.delta) + np.sqrt(self.lambda_) * self.S)
        beta_t = beta_t**2

        # find the point with maximum l2 norm
        ## first step, grid search
        lambda1, lambda2, U = get_factorization(self.V_t)
        t = np.linspace(0, np.pi * 2, 10)
        x1 = np.expand_dims(np.sqrt(beta_t) / np.sqrt(lambda1) * np.sin(t), axis=0)
        x2 = np.expand_dims(np.sqrt(beta_t) / np.sqrt(lambda2) * np.cos(t), axis=0)
        x = np.concatenate([x1, x2], axis=0)
        x = U @ x + theta_hat_t[:, np.newaxis]
        x_l2norm = np.linalg.norm(x, axis=0)
        maxindex = np.argmax(x_l2norm)

        ## second step, use the result from gird search as the initial point of optimizer
        f = lambda t: -np.linalg.norm(get_xy(lambda1=lambda1, lambda2=lambda2, U=U, x0=theta_hat_t, beta=beta_t, t=t))
        res = minimize(fun=f, x0=maxindex)
        maxindex_scipy = res.x[0]
        tilde_theta = U @ np.array(get_xy(lambda1=lambda1, lambda2=lambda2, U=U, x0=theta_hat_t, beta=beta_t, t=maxindex_scipy))

        return tilde_theta


#%% unit test 1, test the algorithm that can find the point
# ## visualize a ellipsoid, $(x-x0)^TA(x-x0)=\beta$
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

# d = 2
# np.random.seed(12345)
# # A = np.random.uniform(size=(d, d))
# # A = A @ A.T
# A = np.array(
#     [
#         [
#             1.0 / 36.0,
#             0.0,
#         ],
#         [0.0, 1.0],
#     ]
# )
# # A = np.array([[3.0, 1.0], [1.0, 1.0]])
# lambda1, lambda2, U = get_factorization(A)
# # print(P.T @ J @ P)
# print(A)
# print(lambda1, lambda2)
# print(U @ np.array([[lambda1, 0.0], [0.0, lambda2]]) @ U.T)
# # x0 = np.random.uniform(size=(d)) * 10
# x0 = np.array([0.0, 10.0])
# # beta = np.random.uniform(low=0.0, high=1.0)
# beta = 1.0

# t = np.linspace(0, np.pi * 2, 10)
# x1 = np.expand_dims(np.sqrt(beta) / np.sqrt(lambda1) * np.sin(t), axis=0)
# x2 = np.expand_dims(np.sqrt(beta) / np.sqrt(lambda2) * np.cos(t), axis=0)
# # np.newaxis
# x = np.concatenate([x1, x2], axis=0)
# x = U @ x + x0[:, np.newaxis]

# plt.figure()
# plt.plot(x[0, :], x[1, :])
# l2norm = np.linalg.norm(x, axis=0)
# maxpoint = np.argmax(l2norm)
# plt.scatter(x[0, maxpoint], x[1, maxpoint])
# plt.xlim((-15, 15))
# plt.ylim((-15, 15))
# plt.grid(True, which="both")

# f = lambda t: -np.linalg.norm(U @ np.array([np.sqrt(beta) / np.sqrt(lambda1) * np.sin(t), np.sqrt(beta) / np.sqrt(lambda2) * np.cos(t)]) + x0)
# res = minimize(fun=f, x0=maxpoint)
# maxpoint_scipy = res.x[0]
# point = U @ np.array([np.sqrt(beta) / np.sqrt(lambda1) * np.sin(maxpoint_scipy), np.sqrt(beta) / np.sqrt(lambda2) * np.cos(maxpoint_scipy)]) + x0
# plt.scatter(point[0], point[1], c="red")
# print(point)
# plt.show()

#%% unit test 2
# v = np.array([2.0, 1.0])
# print(v[:, np.newaxis] @ v[np.newaxis, :])

#%% unit test 3
