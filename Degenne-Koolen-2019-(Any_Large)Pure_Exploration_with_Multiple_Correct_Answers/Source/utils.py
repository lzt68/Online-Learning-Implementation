"""
Python program for golden section search.  This implementation
does not reuse function evaluations and assumes the minimum is c
or d (not on the edges at a or b)
"""

import math

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi


# def gss(f, a, b, xtolerance=1e-5, ftoloerance=1e-5):
#     """
#     Golden-section search
#     to find the minimum of f on [a,b]

#     * f: a strictly unimodal function on [a,b]

#     Example:
#     >>> def f(x): return (x - 2) ** 2
#     >>> x = gss(f, 1, 5)
#     >>> print(f"{x:.5f}")
#     2.00000

#     """
#     while b - a > xtolerance:
#         c = b - (b - a) * invphi
#         d = a + (b - a) * invphi
#         fc = f(c)
#         fd = f(d)
#         if math.fabs(fc - fd) < ftoloerance:
#             return (b + a) / 2, f((b + a) / 2)

#         if f(c) < f(d):
#             b = d
#         else:  # f(c) > f(d) to find the maximum
#             a = c

#     return (b + a) / 2, f((b + a) / 2)


def gss(f, a, b, tolerance=1e-5):
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
    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
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
mu = np.sort(mu)[::-1]
mu0 = mu[-1] - 1  # smallest value minus 1
N0 = np.random.binomial(n=100, p=0.5)
Nt = np.random.binomial(n=100, p=0.5, size=K)

# N0 = 1
# Nt = np.ones(K)


f = lambda x: N0 * (x - mu0) ** 2 + np.sum(Nt * (np.maximum(mu - x, 0) ** 2))

optx, _ = gss(f, mu0, np.max(mu))
print(optx)

optx_alternative = (np.sum(mu * Nt) + mu0 * N0) / (np.sum(Nt) + N0)
print(optx_alternative)

# another way to search for the minimum point
mu_times_Nt = mu * Nt
Nt_cumsum = np.cumsum(Nt)
total_largest_sum = np.cumsum(mu_times_Nt)
center_point = total_largest_sum / Nt_cumsum
## half section search
leftarm = 0
rightarm = K - 1
count_iteration = 0
while True:
    # if rightarm == K:
    #     right_mean = mu0
    #     right_pulling = N0
    # else:
    #     right_mean = center_point[rightarm]
    #     right_pulling = N0

    middle_arm = (leftarm + rightarm) // 2

    new_center_point = (center_point[middle_arm] * Nt_cumsum[middle_arm] + mu0 * N0) / (
        Nt_cumsum[middle_arm] + N0
    )

    if new_center_point > mu[middle_arm + 1] and new_center_point < mu[middle_arm]:
        print(new_center_point)
        break
    elif new_center_point < mu[middle_arm + 1]:
        leftarm = middle_arm + 1
        count_iteration += 1
    elif new_center_point > mu[middle_arm]:
        rightarm = middle_arm
        count_iteration += 1


# plot figure
x_ = np.linspace(start=mu0, stop=np.max(mu), num=1000)
y_ = np.array([f(x) for x in x_])
plt.plot(x_, y_)
plt.axvline(x=optx, color="r", linestyle="--")
for arm in range(K):
    plt.axvline(x=mu[arm], color="b", linestyle="-")
plt.axvline(x=mu0, color="g", linestyle="--")
plt.show()
