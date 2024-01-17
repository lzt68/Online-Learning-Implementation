from typing import Union
import numpy as np


class Conservative_UCB(object):
    def __init__(
        self,
        K: int,
        mu0: Union[np.float64, int, float],
        delta: Union[np.float64, int, float],
        alpha: Union[np.float64, int, float],
    ) -> None:
        """Implement the Conservative UCB algorithm

        Args:
            K (int): Number of alternative arms
            mu0 (Union[np.float64, int, float]): Mean reward of the default arm.
            delta (Union[np.float64, int, float]): Probability threhsold for the incorrect prediction or the violation.
            alpha (Union[np.float64, int, float]): Safety threshold.
        """
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        assert alpha >= 0.0 and alpha <= 1.0, "delta is not in [0, 1]"
        assert mu0 > 0.0 and mu0 < 1.0, "mu0 is not in (0, 1)"

        self.K = K
        self.delta = delta
        self.alpha = alpha
        self.mu0 = mu0

        self.t = 1
        self.action_ = list()
        self.total_reward_ = np.zeros(K + 1)  # arm=0, 1, 2, ..., K
        self.pulling_times_ = np.zeros(K + 1)
        self.mean_reward_ = np.zeros(K + 1)

    def action(self):
        # calculate the length of confidence interval
        Delta_t = np.sqrt(self.rad_1(self.pulling_times_[1:]) / np.maximum(self.pulling_times_[1:], 1))

        # calculate the lower bound for all the arms
        lambda_t = np.zeros(self.K + 1)
        lambda_t[0] = self.mu0
        lambda_t[1:] = self.mean_reward_[1:] - Delta_t
        lambda_t[1:] = np.maximum(lambda_t[1:], 0)

        # calculate the upper bound for all the arms
        theta_t = np.zeros(self.K + 1)
        theta_t[0] = self.mu0
        theta_t[1:] = self.mean_reward_[1:] + Delta_t

        # the arms with highest upper confidence bound
        J_t = np.argmax(theta_t)

        # calculate the lower bound of the cumulative regret
        xi_t = np.sum(self.pulling_times_ * lambda_t) + lambda_t[J_t] - (1 - self.alpha) * self.t * self.mu0
        if xi_t >= 0:
            action = J_t
        else:
            action = 0

        self.action_.append(action)
        return action

    def observe(self, reward):
        self.t += 1
        pulling = self.pulling_times_[self.action_[-1]]
        mean_r = self.mean_reward_[self.action_[-1]]
        self.mean_reward_[self.action_[-1]] = pulling / (pulling + 1) * mean_r + reward / (pulling + 1)
        self.pulling_times_[self.action_[-1]] += 1
        self.total_reward_[self.action_[-1]] += reward

    def rad_1(self, s: Union[np.int32, np.int64, int, np.ndarray]):
        """calculate the numerator of the half of the length

        Args:
            s (Union[np.float64, float, np.ndarray]): s is the pulling times. It can be an interger or an array
        """
        psi_s = 2 * np.log(self.K * (np.maximum(s, 1) ** 3) / self.delta)
        return psi_s


class Conservative_UCB_rad2(object):
    def __init__(
        self,
        K: int,
        mu0: Union[np.float64, int, float],
        delta: Union[np.float64, int, float],
        alpha: Union[np.float64, int, float],
    ) -> None:
        """Implement the Conservative UCB algorithm, with a smaller length of interval.
        The expression is in euqation (9) of Conservative Bandits.

        Args:
            K (int): Number of alternative arms
            mu0 (Union[np.float64, int, float]): Mean reward of the default arm.
            delta (Union[np.float64, int, float]): Probability threhsold for the incorrect prediction or the violation.
            alpha (Union[np.float64, int, float]): Safety threshold.
        """
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        assert alpha >= 0.0 and alpha <= 1.0, "delta is not in [0, 1]"
        assert mu0 > 0.0 and mu0 < 1.0, "mu0 is not in (0, 1)"

        self.K = K
        self.delta = delta
        self.alpha = alpha
        self.mu0 = mu0

        self.t = 1
        self.action_ = list()
        self.total_reward_ = np.zeros(K + 1)  # arm=0, 1, 2, ..., K
        self.pulling_times_ = np.zeros(K + 1)
        self.mean_reward_ = np.zeros(K + 1)

    def action(self):
        # calculate the length of confidence interval
        Delta_t = np.sqrt(self.rad_2(self.pulling_times_[1:]) / np.maximum(self.pulling_times_[1:], 1))

        # calculate the lower bound for all the arms
        lambda_t = np.zeros(self.K + 1)
        lambda_t[0] = self.mu0
        lambda_t[1:] = self.mean_reward_[1:] - Delta_t
        lambda_t[1:] = np.maximum(lambda_t[1:], 0)

        # calculate the upper bound for all the arms
        theta_t = np.zeros(self.K + 1)
        theta_t[0] = self.mu0
        theta_t[1:] = self.mean_reward_[1:] + Delta_t

        # the arms with highest upper confidence bound
        J_t = np.argmax(theta_t)

        # calculate the lower bound of the cumulative regret
        xi_t = np.sum(self.pulling_times_ * lambda_t) + lambda_t[J_t] - (1 - self.alpha) * self.t * self.mu0
        if xi_t >= 0:
            action = J_t
        else:
            action = 0

        self.action_.append(action)
        return action

    def observe(self, reward):
        self.t += 1
        pulling = self.pulling_times_[self.action_[-1]]
        mean_r = self.mean_reward_[self.action_[-1]]
        self.mean_reward_[self.action_[-1]] = pulling / (pulling + 1) * mean_r + reward / (pulling + 1)
        self.pulling_times_[self.action_[-1]] += 1
        self.total_reward_[self.action_[-1]] += reward

    def rad_2(self, s: Union[np.int32, np.int64, int, np.ndarray]):
        """calculate the numerator of the half of the length

        Args:
            s (Union[np.float64, float, np.ndarray]): s is the pulling times. It can be an interger or an array
        """
        zeta = self.K / self.delta
        psi_s = (
            np.log(np.maximum(3, np.log(zeta)))
            + np.log(2 * (np.e**2) * zeta)
            + zeta
            * (1 + np.log(zeta))
            / (zeta - 1)
            / np.log(zeta)
            * np.maximum(np.log(np.maximum(np.log(1 + s), 1)), 1)
        )
        return psi_s


class Conservative_UCB_Overall_Lower_Bound(object):
    def __init__(
        self,
        K: int,
        mu0: Union[np.float64, int, float],
        delta: Union[np.float64, int, float],
        alpha: Union[np.float64, int, float],
    ) -> None:
        """Improve the Conservative UCB algorithm. When we test wether there is possible violation, we don't use the summation of lower bounds,
        but use the lower bound for the total reward.

        Args:
            K (int): Number of alternative arms
            mu0 (Union[np.float64, int, float]): Mean reward of the default arm.
            delta (Union[np.float64, int, float]): Probability threhsold for the incorrect prediction or the violation.
            alpha (Union[np.float64, int, float]): Safety threshold.
        """
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"
        assert alpha >= 0.0 and alpha <= 1.0, "delta is not in [0, 1]"
        assert mu0 > 0.0 and mu0 < 1.0, "mu0 is not in (0, 1)"

        self.K = K
        self.delta = delta
        self.alpha = alpha
        self.mu0 = mu0

        self.t = 1
        self.action_ = list()
        self.total_reward_ = np.zeros(K + 1)  # arm=0, 1, 2, ..., K
        self.pulling_times_ = np.zeros(K + 1)
        self.mean_reward_ = np.zeros(K + 1)

    def action(self):
        # calculate the length of confidence interval
        Delta_t = np.sqrt(self.rad_2(self.pulling_times_[1:]) / np.maximum(self.pulling_times_[1:], 1))

        # calculate the upper bound for all the arms
        theta_t = np.zeros(self.K + 1)
        theta_t[0] = self.mu0
        theta_t[1:] = self.mean_reward_[1:] + Delta_t

        # the arms with highest upper confidence bound
        J_t = np.argmax(theta_t)

        # calculate the lower bound for all the arms
        if J_t == 0:
            lambda_t = self.mu0
        else:
            lambda_t = np.maximum(self.mean_reward_[J_t] - Delta_t[J_t - 1], 0)

        # calculate the half of the length of the cumulative reward
        C_all_t = self.rad_overall()

        # calculate the lower bound of the cumulative regret
        xi_t = np.maximum(np.sum(self.total_reward_) - C_all_t, 0) + lambda_t - (1 - self.alpha) * self.t * self.mu0
        if xi_t >= 0:
            action = J_t
        else:
            action = 0

        self.action_.append(action)
        return action

    def observe(self, reward):
        self.t += 1
        pulling = self.pulling_times_[self.action_[-1]]
        mean_r = self.mean_reward_[self.action_[-1]]
        self.mean_reward_[self.action_[-1]] = pulling / (pulling + 1) * mean_r + reward / (pulling + 1)
        self.pulling_times_[self.action_[-1]] += 1
        self.total_reward_[self.action_[-1]] += reward

    def rad_2(self, s: Union[np.int32, np.int64, int, np.ndarray]):
        """calculate the numerator of the half of the length

        Args:
            s (Union[np.float64, float, np.ndarray]): s is the pulling times. It can be an interger or an array
        """
        zeta = self.K / self.delta
        psi_s = (
            np.log(np.maximum(3, np.log(zeta)))
            + np.log(2 * (np.e**2) * zeta)
            + zeta
            * (1 + np.log(zeta))
            / (zeta - 1)
            / np.log(zeta)
            * np.maximum(np.log(np.maximum(np.log(1 + s), 1)), 1)
        )
        return psi_s

    def rad_overall(self):
        """
        The original length of interval is
        $$
            C_{all, \{n_a\}_{a=1}^K}=(1+\sqrt{\varepsilon})\sqrt{2(1+\varepsilon)\left(\sum_{a=1}^K n_a\right) \log\frac{K\log(1+\varepsilon)\left(\sum_{a=1}^K n_a\right)}{\delta}}.
        $$
        And we take $\epsilon = 1$. In that case, the concentration inequality holds with prob $1-3(\frac{1}{\log w})^2\delta > 1-7\delta$. Thus in the following, I implement the interval as
        $$
            C_{all, \{n_a\}_{a=1}^K}=(1+\sqrt{\varepsilon})\sqrt{2(1+\varepsilon)\left(\sum_{a=1}^K n_a\right) \log\frac{7K\log(1+\varepsilon)\left(\sum_{a=1}^K n_a\right)}{\delta}}.
        $$
        """
        e = 1
        pulling_alter = np.maximum(np.sum(self.pulling_times_[1:]), 1)
        length_interval = (1 + np.sqrt(e)) * np.sqrt(
            2 * (1 + e) * pulling_alter * np.log(7 * self.K * np.log((1 + e) * pulling_alter) / self.delta)
        )
        return length_interval


# %% unit test 1, test wether the we can run the loop
# from env import Env_Gaussian_Fixedmu0

# random_seed = 1
# np.random.seed(random_seed)

# K = 4
# n = 1000
# r_list = np.random.uniform(low=0.0, high=1.0, size=K)
# mu0 = 0.5
# delta = 0.1

# env = Env_Gaussian_Fixedmu0(K=K, mu0=mu0, r_list=r_list, random_seed=random_seed, n=n)
# agent = Conservative_UCB(K=K, mu0=mu0, alpha=1.0, delta=delta)
# reward_ = list()
# while not env.if_stop():
#     arm = agent.action()
#     reward = env.response(arm=arm)
#     agent.observe(reward=reward)
#     reward_.append(reward)
# print(f"True real reward is {r_list}")
# print(f"Pulling times of each arm {agent.pulling_times_}")
# print(f"Observed mean reward is {agent.mean_reward_}")

# %% unit test 2, test wether the we can observe sublinear regret
# from env import Env_Gaussian_Fixedmu0

# random_seed = 1
# np.random.seed(random_seed)

# K = 4
# r_list = np.random.uniform(low=0.0, high=1.0, size=K)
# mu0 = 0.5
# delta = 0.1

# ratio = list()

# n_exp = 10
# for n in range(100, 1000, 100):
#     regret_ = []
#     for exp_id in range(n_exp):
#         env = Env_Gaussian_Fixedmu0(K=K, mu0=mu0, r_list=r_list, random_seed=exp_id, n=n)
#         agent = Conservative_UCB(K=K, mu0=mu0, alpha=0.5, delta=delta)
#         reward_ = list()
#         while not env.if_stop():
#             arm = agent.action()
#             reward = env.response(arm=arm)
#             agent.observe(reward=reward)

#         total_reward = np.sum([r_list[action - 1] if action >= 1 else mu0 for action in agent.action_])
#         best_reward = n * np.maximum(np.max(r_list), mu0)
#         regret = best_reward - total_reward
#         regret_.append(regret)

#     ratio.append(np.mean(regret_) / n)
# print(ratio)

# %% unit test 3, test wether the we can run the loop with a smaller loop
# from env import Env_Gaussian_Fixedmu0

# random_seed = 1
# np.random.seed(random_seed)

# K = 4
# n = 1000
# r_list = np.random.uniform(low=0.0, high=1.0, size=K)
# mu0 = 0.5
# delta = 0.1

# env = Env_Gaussian_Fixedmu0(K=K, mu0=mu0, r_list=r_list, random_seed=random_seed, n=n)
# agent = Conservative_UCB_rad2(K=K, mu0=mu0, alpha=1.0, delta=delta)
# reward_ = list()
# while not env.if_stop():
#     arm = agent.action()
#     reward = env.response(arm=arm)
#     agent.observe(reward=reward)
#     reward_.append(reward)
# print(f"True real reward is {r_list}")
# print(f"Pulling times of each arm {agent.pulling_times_}")
# print(f"Observed mean reward is {agent.mean_reward_}")

# %% unit test 4, test the algorithm which consider lower bound for the cumulative reward
# from env import Env_Gaussian_Fixedmu0

# random_seed = 1
# np.random.seed(random_seed)

# K = 50
# n = 5000
# r_list = np.random.uniform(low=0.0, high=1.0, size=K)
# mu0 = 0.5
# delta = 0.1

# env = Env_Gaussian_Fixedmu0(K=K, mu0=mu0, r_list=r_list, random_seed=random_seed, n=n)
# agent = Conservative_UCB(K=K, mu0=mu0, alpha=0.5, delta=delta)
# reward_ = list()
# while not env.if_stop():
#     arm = agent.action()
#     reward = env.response(arm=arm)
#     agent.observe(reward=reward)
#     reward_.append(reward)
# print(f"True real reward is {r_list}")
# print(f"Pulling times of each arm {agent.pulling_times_}")
# print(f"Observed mean reward is {agent.mean_reward_}")

# env = Env_Gaussian_Fixedmu0(K=K, mu0=mu0, r_list=r_list, random_seed=random_seed, n=n)
# agent = Conservative_UCB_Overall_Lower_Bound(K=K, mu0=mu0, alpha=0.5, delta=delta)
# reward_ = list()
# while not env.if_stop():
#     arm = agent.action()
#     reward = env.response(arm=arm)
#     agent.observe(reward=reward)
#     reward_.append(reward)
# print(f"True real reward is {r_list}")
# print(f"Pulling times of each arm {agent.pulling_times_}")
# print(f"Observed mean reward is {agent.mean_reward_}")

# %% unit test 5, try to conduct numeric experiments
# import numpy as np
# import pandas as pd
# from env import Env_Gaussian_Fixedmu0

# record = pd.DataFrame(
#     columns=["K", "n", "mu0", "r_list", "algorithm", "alpha", "delta", "Total_Reward", "Regret", "ratio"]
# )

# K_ = np.array([5, 10, 20])  # number of arms
# # n_over_K_ = np.arange(10, 35, 10) # here we determine the total number of rounds as K * c,
# n_over_K = 10
# alg_class = [Conservative_UCB, Conservative_UCB_rad2, Conservative_UCB_Overall_Lower_Bound]
# mu0 = 0.6
# alpha = 1 / 6  # the safety threshold is 0.5
# delta = 0.1

# n_exp = 5
# for K in K_:
#     n = K * n_over_K
#     for alg in alg_class:
#         total_regert__ = np.zeros((n_exp, n))
#         total_reward__ = np.zeros((n_exp, n))
#         ratio__ = np.zeros((n_exp, n))

#         for exp_id in range(n_exp):
#             regret_ = np.zeros(n)
#             reward_ = np.zeros(n)

#             r_list = np.ones(K) * 0.65
#             r_list[0] = 0.7
#             best_reward = np.maximum(np.max(r_list), mu0)

#             # shuffle the arms
#             np.random.seed(exp_id)
#             permuted_index = np.arange(K)
#             np.random.shuffle(permuted_index)
#             r_list = r_list[permuted_index]

#             env = Env_Gaussian_Fixedmu0(K=K, mu0=mu0, r_list=r_list, random_seed=exp_id, n=n)
#             agent = alg(K=K, mu0=mu0, alpha=alpha, delta=delta)
#             while not env.if_stop():
#                 arm = agent.action()
#                 reward = env.response(arm=arm)
#                 agent.observe(reward=reward)

#                 if arm == 0:
#                     reward_[env.t - 1] = mu0
#                     regret_[env.t - 1] = best_reward - mu0
#                 else:
#                     reward_[env.t - 1] = r_list[arm - 1]
#                     regret_[env.t - 1] = best_reward - r_list[arm - 1]

#             total_regert_ = np.cumsum(regret_)
#             total_reward_ = np.cumsum(reward_)
#             ratio_ = total_regert_ / np.arange(1, n + 1, 1)

#             total_regert__[exp_id, :] = total_regert_
#             total_reward__[exp_id, :] = total_reward_
#             ratio__[exp_id, :] = ratio_

#         # safe the numeric record
#         record.loc[record.shape[0]] = np.array(
#             [
#                 K,
#                 n,
#                 mu0,
#                 np.array2string(r_list, threshold=11e3),
#                 alg.__name__,
#                 alpha,
#                 delta,
#                 np.array2string(total_reward__, threshold=11e3),
#                 np.array2string(total_regert__, threshold=11e3),
#                 np.array2string(ratio__, threshold=11e3),
#             ],
#             dtype=object,
#         )
