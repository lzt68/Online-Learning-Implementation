import numpy as np


# notations and functions in the paper
def d_fun(x, y):
    x = np.maximum(0.0001, x)
    x = np.minimum(0.9999, x)
    y = np.maximum(0.0001, y)
    y = np.minimum(0.9999, y)
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


def Get_w_star(mu: np.ndarray):
    """Given the array of mean reward, solve the optimization problem and get the optimal pulling fraction
    The optimization target is
    $$
        w^*(\mu)=\arg\max_{w\in\Sigma_K} \min_{a\ne 1}(w_1+w_a)I_{\frac{w_1}{w_1+w_a}}(\mu_1, \mu_a)
    $$
    where $I_{\alpha}(\mu_1, \mu_2)=\alpha d\left(\mu_1, \alpha\mu_1+(1-\alpha)\mu_2\right)+(1-\alpha)d(\mu_2, \alpha\mu_1+(1-\alpha)\mu_2)$
    and $d(x,y)=x\log\frac{x}{y}+(1-x)\log\frac{1-x}{1-y}$

    Args:
        mu (np.ndarray): Array of mean rewards
    """
    K = mu.shape[0]

    # sort the array of mean rewards to make sure decreasing order, switch back to the sequence when we get the weights
    index = np.argsort(mu)[::-1]
    mu_temp = mu.copy()
    mu_temp = mu_temp[index]

    if np.abs(mu_temp[0] - mu_temp[1]) < 1e-6:
        # if \hat{\mu}_1 is equal to \hat{\mu}_2, we return uniform sampling weight
        return np.ones(K) / K

    # use bisection to find the y^* such that F(y^*) = 1
    epsilon_x = 0.001
    epsilon_fun = 0.01
    left = 0
    right = d_fun(mu_temp[0], mu_temp[1]) / 2
    while F_fun(right, mu=mu_temp, K=K) < 1:
        left = right
        right = (d_fun(mu_temp[0], mu_temp[1]) + right) / 2
    temp = (left + right) / 2

    count = 0
    while (np.abs(right - left) > epsilon_x) or (np.abs(F_fun(temp, mu=mu_temp, K=K) - 1) > epsilon_fun):
        if F_fun(temp, mu=mu_temp, K=K) >= 1:
            right = temp
        else:
            left = temp
        temp = (left + right) / 2
        count += 1
    y_star = (left + right) / 2

    # calculate the optimal pulling fraction
    x_a_y_star = np.array([1.0] + [x_fun(a=aa, y=y_star, mu=mu_temp) for aa in range(2, K + 1)])
    w_star = x_a_y_star / np.sum(x_a_y_star)

    # switch back to the original sequence
    index_back = np.zeros(K)
    index_back[index] = np.arange(K)
    w_star = w_star[index_back.astype(int)]

    return w_star


def Get_T_star(mu):
    mu = np.sort(mu)[::-1]

    # get the w_star
    w_star = Get_w_star(mu=mu)

    # calculate T_star
    lambda_1_2 = (w_star[0] * mu[0] + w_star[1] * mu[1]) / (w_star[0] + w_star[1])
    T_star = 1 / (w_star[0] * d_fun(mu[0], lambda_1_2) + w_star[1] * d_fun(mu[1], lambda_1_2))

    return T_star


def Get_Lower_Bound_pulling(mu, delta):
    T_star = Get_T_star(mu=mu)
    return T_star * d_fun(delta, 1 - delta)


# %% unit test 1, test whether we can get same g_a(x_a^*) for all a
# K = 4
# mu = np.array([0.5, 0.2, 0.3, 0.4])
# w_star = Get_w_star(mu=mu)

# for aa in range(1, K):
#     lambda_1_a = (w_star[0] * mu[0] + w_star[aa] * mu[aa]) / (w_star[0] + w_star[aa])
#     T_star_a = 1 / (w_star[0] * d_fun(mu[0], lambda_1_a) + w_star[aa] * d_fun(mu[aa], lambda_1_a))
#     print(f"arm index {aa+1}, T_star {T_star_a}")

# %% unit test 2, changing speed of $kl(\delta,1-\delta)$
# for delta in np.linspace(start=0.5, stop=0.01, num=10):
#     print(f"delta {delta}, kl(delta, 1-delta): {d_fun(delta, 1-delta)}")
