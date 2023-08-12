import numpy as np


# notations and functions in the paper
def d_fun(x, y):
    x = np.maximum(0.001, x)
    x = np.minimum(0.999, x)
    y = np.maximum(0.001, y)
    y = np.minimum(0.999, y)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def I(mu1, mu2, alpha):
    mu_temp = alpha * mu1 + (1 - alpha) * mu2
    return alpha * d_fun(mu1, mu_temp) + (1 - alpha) * d_fun(mu2, mu_temp)


def g(a, x, mu):
    alpha = 1 / (1 + x)
    return (1 + x) * I(mu[0], mu[a - 1], alpha)


def x_fun(a, y, mu, epsilon=0.001):
    # a is in {1, 2, ..., K}
    left = 0
    right = 100
    while g(a, right, mu=mu) < y:
        left = right
        right *= 2
    while np.abs(right - left) > epsilon:
        temp = (left + right) / 2
        if g(a, temp, mu=mu) >= y:
            right = temp
        else:
            left = temp
    return (left + right) / 2


def F_fun(y, mu, K):
    ratio_sum = 0
    for aa in range(2, K + 1):
        x_a_y = x_fun(a=aa, y=y, mu=mu)
        if np.abs(mu[0] - mu[aa - 1]) < 1e-6:
            ratio_sum += x_a_y**2
        else:
            temp_mu = (mu[0] + x_a_y * mu[aa - 1]) / (1 + x_a_y)
            ratio_sum += d_fun(mu[0], temp_mu) / d_fun(mu[aa - 1], temp_mu)
    return ratio_sum
