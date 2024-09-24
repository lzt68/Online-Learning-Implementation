"""
Python program for golden section search.  This implementation
does not reuse function evaluations and assumes the minimum is c
or d (not on the edges at a or b)
"""

import math

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi


def gss(f, a, b, xtolerance=1e-5, ftoloerance=1e-5):
    """
    Golden-section search
    to find the minimum of f on [a,b]

    * f: a strictly unimodal function on [a,b]

    Example:
    >>> def f(x): return (x - 2) ** 2
    >>> x = gss(f, 1, 5)
    >>> print(f"{x:.5f}")
    2.00000

    """
    while b - a > xtolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        fc = f(c)
        fd = f(d)
        if math.fabs(fc - fd) < ftoloerance:
            return (b + a) / 2, f((b + a) / 2)

        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2, f((b + a) / 2)


# %% unit test 1
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)
K = 5
mu = np.random.uniform(low=0.0, high=1.0, size=K)
mu = np.sort(mu)
mu0 = mu[0] - 1
N0 = np.random.binomial(n=100, p=0.5)
Nt = np.random.binomial(n=100, p=0.5, size=K)
f = lambda x: np.sum(N0 * (x - mu0) ** 2 + Nt * (np.maximum(mu - x, 0) ** 2))

optx, _ = gss(f, mu0, mu[-1])

# # plot figure
# x_ = np.linspace(start=mu0, stop=mu[-1], num=1000)
# y_ = np.array([f(x) for x in x_])
# plt.plot(x_, y_)
# plt.axvline(x=optx, color="r", linestyle="--")
# plt.show()
