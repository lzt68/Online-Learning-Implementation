import numpy as np
from numpy import linalg
from copy import deepcopy


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

        return lambda1, lambda2, U


class ConfidenceBall_Agent(object):
    def __init__(self) -> None:
        pass


class OFUL_Agent(object):
    def __init__(
        self,
        R: float,
        lambda_: float,
        L: float,
        S: float,
    ) -> None:
        """Initialization of OFUL algorithm,
        O: Optimism, U: Uncertainty, F: Face, L: Linear
        We assume the action space is always the unit ball in 2-d space

        Args:
            R (float): The nois is conditionally R-subgaussian random variable
            _lambda (float): Regularization coefficient
            L (float): \|X_t\|_2 \le L
            S (float): \|\theta_*\|_2 \le S
        """
        self.d = d
        self.R = R
        self.lambda_ = lambda_
        self.L = L
        self.S = S

        self.V = np.eye(d) * lambda_
        self.H_action = list()  # history of reward
        self.H_reward = list()  # history of action

        self.t = 0

    def action(self):
        theta = self.get_theta()
        theta_l2 = linalg.norm(theta, ord=2)
        if np.abs(theta_l2) < 1e-4:
            act = np.zeros(self.d)
        else:
            act = theta / theta_l2
        return act

    def update(self, reward):
        self.t += 1

    def get_theta(self):
        # solve the problem
        # \tilde{\theta}_t=\arg\max_{\theta\in C_{t-1}}\|\theta\|_2
        return np.zeros(self.d)

    def get_beta(self):
        # calculate the beta_t
        pass


# unit test 1, test the algorithm that can find the point
## visualize a ellipsoid, $(x-x0)^TA(x-x0)=\beta$
import matplotlib.pyplot as plt
from scipy.optimize import minimize

d = 2
np.random.seed(12345)
A = np.random.uniform(size=(d, d))
A = A @ A.T
# A = np.array([[3.0, 1.0], [1.0, 1.0]])
lambda1, lambda2, U = get_factorization(A)
# print(P.T @ J @ P)
print(A)
print(lambda1, lambda2)
print(U @ np.array([[lambda1, 0.0], [0.0, lambda2]]) @ U.T)
x0 = np.random.uniform(size=(d)) * 10
beta = np.random.uniform(low=0.0, high=1.0)

t = np.linspace(0, np.pi * 2, 10)
x1 = np.expand_dims(np.sqrt(beta) / np.sqrt(lambda1) * np.sin(t), axis=0)
x2 = np.expand_dims(np.sqrt(beta) / np.sqrt(lambda2) * np.cos(t), axis=0)
np.newaxis
x = np.concatenate([x1, x2], axis=0)
x = U @ x + x0[:, np.newaxis]

plt.figure()
plt.plot(x[0, :], x[1, :])
l2norm = np.linalg.norm(x, axis=0)
maxpoint = np.argmax(l2norm)
plt.scatter(x[0, maxpoint], x[1, maxpoint])
plt.xlim((-15, 15))
plt.ylim((-15, 15))
plt.grid(True, which="both")

f = lambda t: -np.linalg.norm(U @ np.array([np.sqrt(beta) / np.sqrt(lambda1) * np.sin(t), np.sqrt(beta) / np.sqrt(lambda2) * np.cos(t)]) + x0)
res = minimize(fun=f, x0=maxpoint)
maxpoint_scipy = res.x[0]
point = U @ np.array([np.sqrt(beta) / np.sqrt(lambda1) * np.sin(maxpoint_scipy), np.sqrt(beta) / np.sqrt(lambda2) * np.cos(maxpoint_scipy)]) + x0
plt.scatter(point[0], point[1], c="red")

plt.show()
