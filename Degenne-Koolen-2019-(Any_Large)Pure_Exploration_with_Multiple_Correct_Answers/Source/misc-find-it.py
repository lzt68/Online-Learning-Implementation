# %% golden section function
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
# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(12345)
# K = 4000
# # mu = np.random.uniform(low=0.0, high=1.0, size=K)
# mu = np.random.normal(loc=0.0, scale=100, size=K)
# mu = np.sort(mu)[::-1]
# mu0 = mu[-1] - 100  # smallest value minus 1
# N0 = np.random.binomial(n=100, p=0.5)
# Nt = np.random.binomial(n=100, p=0.5, size=K)

# # N0 = 1
# # Nt = np.ones(K)


# f = lambda x: N0 * (x - mu0) ** 2 + np.sum(Nt * (np.maximum(mu - x, 0) ** 2))

# optx, _ = gss(f, mu0, np.max(mu))
# print(optx)

# # optx_alternative = (np.sum(mu * Nt) + mu0 * N0) / (np.sum(Nt) + N0)
# # print(optx_alternative)

# # another way to search for the minimum point
# mu_times_Nt = mu * Nt
# Nt_cumsum = np.cumsum(Nt)
# total_largest_sum = np.cumsum(mu_times_Nt)
# center_point = total_largest_sum / Nt_cumsum
# ## half section search
# leftarm = 0
# rightarm = K - 1
# count_iteration = 0
# while True:
#     # if rightarm == K:
#     #     right_mean = mu0
#     #     right_pulling = N0
#     # else:
#     #     right_mean = center_point[rightarm]
#     #     right_pulling = N0

#     middle_arm = (leftarm + rightarm) // 2

#     new_center_point = (center_point[middle_arm] * Nt_cumsum[middle_arm] + mu0 * N0) / (
#         Nt_cumsum[middle_arm] + N0
#     )

#     middle_next_mean = mu0 if middle_arm == K - 1 else mu[middle_arm + 1]
#     if new_center_point > middle_next_mean and new_center_point < mu[middle_arm]:
#         print(new_center_point)
#         break
#     elif new_center_point < middle_next_mean:
#         leftarm = middle_arm + 1
#         count_iteration += 1
#     elif new_center_point > mu[middle_arm]:
#         rightarm = middle_arm
#         count_iteration += 1

# print(f"total iteration: {count_iteration}")

# # # plot figure
# # x_ = np.linspace(start=mu0, stop=np.max(mu), num=1000)
# # y_ = np.array([f(x) for x in x_])
# # plt.plot(x_, y_)
# # plt.axvline(x=optx, color="r", linestyle="--")
# # for arm in range(K):
# #     plt.axvline(x=mu[arm], color="b", linestyle="-")
# # plt.axvline(x=mu0, color="g", linestyle="--")
# # plt.show()

# %% unit test 2, identify the arm with smallest index that is in It
import numpy as np
import matplotlib.pyplot as plt
from time import time

np.random.seed(12345)
# K = 10
# hatmu = np.random.normal(loc=0.0, scale=1, size=K)
# hatmu[4] = 1.39340583
# hatmu[5] = 1.96578057

# K = 10000
# hatmu = np.random.normal(loc=0.0, scale=1, size=K)
# hatmu = np.sort(hatmu)
# pulling = np.random.binomial(n=500, p=0.5, size=K)

K = 10
hatmu = np.array(
    [
        0.71289112,
        0.73278765,
        0.67111129,
        0.98740475,
        0.60109015,
        0.76363685,
        0.79895482,
        0.67667109,
        0.86822559,
        1.02183551,
    ]
)
pulling = np.array([104.0, 19.0, 19.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0, 18.0])

xi = 0.5
ft = 58.286804682977795

# %% find the optimal, bench mark


def enumerate_benchmark(hatmu, pulling, xi, ft):
    maxmu = np.max(hatmu)
    for arm in range(1, K + 1):
        if hatmu[arm - 1] < xi:
            continue

        if hatmu[arm - 1] == maxmu:
            return arm, maxmu, 0

        N_arm = pulling[arm - 1]
        mu_arm = hatmu[arm - 1]

        Nt_temp = pulling[hatmu > mu_arm]
        mu_temp = hatmu[hatmu > mu_arm]

        f = (
            lambda x: (
                N_arm * (x - mu_arm) ** 2
                + np.sum(Nt_temp * (np.maximum(mu_temp - x, 0) ** 2))
            )
            / 2
        )
        opt, optval = gss(f, mu_arm, np.max(mu_temp))

        if optval < ft:
            return arm, opt, optval


start_time = time()
arm, opt, optval = enumerate_benchmark(hatmu, pulling, xi, ft)
end_time = time()
print("old method:", arm, opt, optval, f"time consumption {end_time- start_time}")

# %% find the optimal, new method
start_time = time()

arms_above_xi = np.where(hatmu > xi)[0] + 1  # O(K)
num_arms_above_xi = len(arms_above_xi)

mu_above_xi = hatmu[arms_above_xi - 1]  # O(K)
pulling_above_xi = pulling[arms_above_xi - 1]  # O(K)
sorted_index_mu_above_xi = np.argsort(mu_above_xi)[::-1]  # O(K log K)

sorted_mu_above_xi = mu_above_xi[sorted_index_mu_above_xi]  # O(K)
# sorted_arm_above_xi = arms_above_xi[sorted_index_mu_above_xi]  # O(K)
sorted_pulling_above_xi = pulling_above_xi[sorted_index_mu_above_xi]  # O(K)
# sorted_mu_above_xi[0] is the maximum value of mu_above_xi
# sorted_mu_above_xi[1] is the second maximum value of mu_above_xi
# sorted_arm_above_xi[0] is the arm index that has the maximum hatmu

mu_order_index = np.argsort(hatmu)[::-1]
arm_order = np.argsort(mu_order_index)
# arm_order[i-1] is the number of arms whose mu is greater than i^th arm

# pre-calculate value
mu_times_Nt = sorted_mu_above_xi * sorted_pulling_above_xi
# Nt_cumsum = np.cumsum(sorted_pulling_above_xi)
cum_sorted_pulling_above_xi = np.cumsum(sorted_pulling_above_xi)
total_largest_sum = np.cumsum(mu_times_Nt)
center_point = total_largest_sum / cum_sorted_pulling_above_xi

# this part is used to calculate the function value
cum_sorted_mu_above_xi = np.cumsum(sorted_mu_above_xi)  # first order sum
# cum_sorted_pulling_above_xi = np.cumsum(sorted_pulling_above_xi)
cum_sorted_pulling_mu_above_xi = np.cumsum(sorted_pulling_above_xi * sorted_mu_above_xi)
cum2_sorted_pulling_mu_above_xi = np.cumsum(
    sorted_pulling_above_xi * sorted_mu_above_xi**2
)

for arm in range(1, K + 1):
    find_or_not = False
    if hatmu[arm - 1] < xi:
        continue
    if arm_order[arm - 1] == 0:
        print("new method: ", arm, hatmu[arm - 1], 0)

    N_arm = pulling[arm - 1]
    mu_arm = hatmu[arm - 1]

    leftindex = 0
    rightindex = arm_order[arm - 1] - 1

    count_num = 0
    while True:
        count_num += 1

        middle_index = (leftindex + rightindex) // 2

        new_center_point = (
            center_point[middle_index] * cum_sorted_pulling_above_xi[middle_index]
            + mu_arm * N_arm
        ) / (cum_sorted_pulling_above_xi[middle_index] + N_arm)

        middle_next_mean = (
            mu_arm
            if middle_index == arm_order[arm - 1] - 1
            else sorted_mu_above_xi[middle_index + 1]
        )
        if (
            new_center_point > middle_next_mean
            and new_center_point < sorted_mu_above_xi[middle_index]
        ):
            # then we need to test whether the function value is below ft
            fval = (
                N_arm + cum_sorted_pulling_above_xi[middle_index]
            ) * new_center_point**2
            fval -= (
                2
                * (N_arm * mu_arm + cum_sorted_pulling_mu_above_xi[middle_index])
                * new_center_point
            )
            fval += N_arm * mu_arm**2 + cum2_sorted_pulling_mu_above_xi[middle_index]
            fval /= 2
            if fval < ft:
                print("new method:", arm, new_center_point, fval)
                find_or_not = True
            break
        elif new_center_point < middle_next_mean:
            leftindex = middle_index + 1
        elif new_center_point > sorted_mu_above_xi[middle_index]:
            rightindex = middle_index

    if find_or_not:
        break
end_time = time()
print(f"time consumption {end_time- start_time}")

# %% for specific mean reward vector, find it
# import numpy as np
# import matplotlib.pyplot as plt
