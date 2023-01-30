import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import norm, multivariate_normal
from typing import Callable
from typing import Union


def MHSampling(N: int, M: int, d: int, g: Callable, random_seed: int = 12345, verbose=False) -> np.ndarray:
    """Implementation of Metropolitan Hastings Sampling

    Args:
        N (int): The beginning number of sampling, for example,
            when N=10, we will record the sampling poing from 10th point
        M (int): Total number of sampling point
        d (int): Dimension of sampling point
        g (Callable): The callable function, which is propotion to the actual density function of distribution.
            Its input is an array whose shape is (d,)
        random_seed (int, optional): Random seed
        verbose (bool, optional): Control whether output the progress of sampling. Defaults to False, no output

    Returns:
        np.ndarray: The list of sampling points, whose shape is (M, d)
    """
    np.random.seed(random_seed)

    x0 = np.zeros(d)
    x = np.zeros(d)
    if verbose:
        print("Warm Up phase")
    for _ in tqdm(range(N - 1), disable=not verbose):
        x = np.random.multivariate_normal(mean=x0, cov=np.eye(N=d))
        if g(x) >= g(x0):
            alpha = 1
        else:
            alpha = g(x) / g(x0)
        if np.random.uniform(low=0.0, high=1.0) < alpha:
            x0 = x

    sampling_points = np.zeros((M, d))
    sampling_points[0, :] = x0
    if verbose:
        print("Sampling phase")

    for count in tqdm(range(1, M), disable=not verbose):
        x = np.random.multivariate_normal(mean=x0, cov=np.eye(N=d))
        if g(x) >= g(x0):
            alpha = 1
        else:
            alpha = g(x) / g(x0)
        if np.random.uniform(low=0.0, high=1.0) < alpha:
            x0 = x
        sampling_points[count, :] = deepcopy(x0)

    return sampling_points


#%% unit test 1
# import matplotlib.pyplot as plt
# # x_ = np.arange(-5, 5, step=0.1)
# # mean = 0.0
# # std = 1.0
# # prob = np.array([normpdf(x=x, mean=mean, Sigma=std) for x in x_])
# # prob_real = norm.pdf(x_)

# # plt.figure()
# # plt.plot(x_, prob, label="My Normal pdf")
# # plt.plot(x_, prob_real, label="Real Normal pdf")
# # plt.legend()
# # plt.show()

# x, y = np.mgrid[-1:1:0.01, -1:1:0.01]
# pos = np.dstack((x, y))
# rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
# # rv = multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contourf(x, y, rv.pdf(pos))
# plt.show()

#%% unit test 2
# import seaborn as sns
# import matplotlib.pyplot as plt

# mu = 1.0
# sigma = 2.0
# # g = lambda x: np.exp(-((x - mu) ** 2) / 2 / sigma**2)
# # g = lambda x: 1.0 / np.exp(2 * x) if x >= 0 else 0.0
# g = lambda x: 2.0 if x <= 1.0 and x >= 0.0 else 0.0
# N = 10000
# M = 5000
# sampling_points = MHSampling(N=N, M=M, d=1, g=g)
# print(sampling_points.shape)
# sns.set_style("whitegrid")
# sns.kdeplot(sampling_points[:, 0], label="approximate")

# x = np.linspace(0, 5, 1000)
# # real_dist = np.exp(-((x - mu) ** 2) / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
# # real_dist = np.array([g(xx) * 2 for xx in x])
# real_dist = np.array([g(xx) / 2 for xx in x])
# plt.plot(x, real_dist, label="real")
# plt.legend()
# plt.show()

#%% unit test 3
# from scipy.stats import beta
# import seaborn as sns
# import matplotlib.pyplot as plt

# a = 200.0
# b = 500.0
# rv = beta(a, b)

# N = 100
# M = 5000
# g = rv.pdf
# sampling_points = MHSampling(N=N, M=M, d=1, g=g, verbose=True)
# sns.set_style("whitegrid")
# sns.kdeplot(sampling_points[:, 0], label="approximate")

# x = np.linspace(0, 5, 1000)
# real_dist = np.array([g(xx) for xx in x])
# plt.plot(x, real_dist, label="real")
# plt.legend()
# plt.show()
