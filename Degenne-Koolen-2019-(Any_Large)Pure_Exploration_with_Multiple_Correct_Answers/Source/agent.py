import numpy as np


def gss(f, a, b, threshold, ftoloerance=1e-3):
    # use golden section search to judege whether function f's minimum point is below threshold
    if f(a) <= threshold or f(b) <= threshold:
        return True

    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    while True:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        fc = f(c)
        fd = f(d)

        if fc <= threshold or fd <= threshold:
            return True

        if np.abs(fc - fd) < ftoloerance:
            return False

        if fc < fd:
            b = d
        else:
            a = c


class Sticky_TaS_old(object):
    # Sticky Track-and-Stop
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        self.delta = delta
        self.K = K
        self.xi = xi

        self.mean_reward_ = np.zeros(K)
        self.sum_pulling_fraction = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.pulling_list = [kk for kk in range(1, K + 1)]

        C = 10  # I am not sure C=10 is enough to fulfill the requirement
        self.beta = lambda x: np.log(C) + 2 * np.log(x) + np.log(1 / delta)
        self.function_f = lambda x: np.log(C) + 10 * np.log(x)

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert not self.stop, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = (
            self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        )
        self.t += 1

        # calculate the arm to be pulled in the next round
        if len(self.pulling_list) == 0:
            # It = self.Get_It(self.mean_reward_, self.pulling_times_)
            # it = It[0]
            # wt = self.Get_wt(self.mean_reward_, it=it)
            wt = self.Get_wt(self.mean_reward_, self.pulling_times_)
            ## C-Track
            epsilon = 1 / np.sqrt(self.K**2 + self.t)
            projected_w = self.get_projection(wt, epsilon)
            self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w
            arm = np.argmax(self.sum_pulling_fraction - self.pulling_times_) + 1
            self.pulling_list.append(arm)

        # determine whether to stop
        max_mean = np.max(self.mean_reward_)
        if max_mean > self.xi:
            a0 = np.argmax(self.mean_reward_) + 1
            beta_t = self.beta(self.t - 1)
            condition = (
                self.pulling_times_[a0 - 1]
                * (self.mean_reward_[arm - 1] - self.xi) ** 2
                / 2
            )
            if beta_t < condition:
                self.stop = True
                return a0
        else:
            for arm in np.arange(1, self.K + 1):
                beta_t = self.beta(self.t - 1)
                condition = (
                    self.pulling_times_[arm - 1]
                    * (self.mean_reward_[arm - 1] - self.xi) ** 2
                    / 2
                )
                if condition <= beta_t:
                    # that means we can find an instance that is in both $\neg i$
                    # and $\mathcal{D}_t$
                    return None
            self.stop = True
            return "No Arms Above xi"

    def if_stop(self):
        return self.stop

    def get_projection(self, w, epsilon):
        # project the w into the $[\epsilon, 1]^K \cap \Sigma_K$, through solving linear optimization problem
        # Please check README.md to see why the following codes can find the projection
        projected_w = np.zeros(self.K)
        threshold_index = w < epsilon
        projected_w[threshold_index] = epsilon

        gap = np.sum(np.maximum(epsilon - w, 0))
        projected_w[~threshold_index] = w[~threshold_index] - gap / (
            np.sum(~threshold_index)
        )

        return projected_w

    def Get_wt(self, hatmu, pulling):
        max_mean = np.max(hatmu)
        if max_mean < self.xi:
            wt = 2 / (hatmu - self.xi) ** 2
            wt = wt / np.sum(wt)
            return wt
        else:  # max_mean \geq self.xi
            It = self.Get_It(hatmu, pulling)
            it = It[0]
            if hatmu[it - 1] > self.xi:
                wt = np.zeros(self.K)
                wt[it - 1] = 1
                return wt
            else:
                wt = np.ones(self.K) / self.K
                return wt

    def Get_It(self, hatmu: np.ndarray, pulling: np.ndarray):
        best_emp_arm = np.argmax(hatmu) + 1
        best_emp = hatmu[best_emp_arm - 1]
        if best_emp < self.xi:
            # $\max_a \hat{\mu}_{a,t} < xi$
            return np.arange(1, self.K + 1)

        ft = self.function_f(self.t - 1)
        mu_temp = hatmu.copy()
        mu_temp[hatmu > self.xi] = self.xi
        condition = np.sum(pulling * (mu_temp - hatmu) ** 2 / 2)
        if condition < ft:
            return np.arange(1, self.K + 1)

        It = list()
        for arm in range(1, self.K + 1):
            if hatmu[arm - 1] > self.xi:
                hat_mu_arm_t = hatmu[arm - 1]
                N_arm_t = pulling[arm - 1]
                index_arm = hatmu > hat_mu_arm_t
                mu_temp = hatmu[index_arm]
                Nt = pulling[index_arm]
                f = lambda x: np.sum(
                    N_arm_t * (x - hat_mu_arm_t) ** 2
                    + Nt * (np.maximum(mu_temp - x, 0) ** 2)
                )
                if gss(f, hatmu[arm - 1], best_emp, ft):
                    It.append(arm)
            else:
                hat_mu_arm_t = hatmu[arm - 1]
                N_arm_t = pulling[arm - 1]
                index_better_than_arm = hatmu > self.xi
                mu_temp = hatmu[index_better_than_arm]
                Nt = pulling[index_better_than_arm]
                f = lambda x: np.sum(
                    N_arm_t * (x - hat_mu_arm_t) ** 2
                    + Nt * (np.maximum(mu_temp - x, 0) ** 2)
                )
                if gss(f, self.xi, best_emp, ft):
                    It.append(arm)
        return It


class Sticky_TaS(object):
    # Sticky Track-and-Stop
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        self.delta = delta
        self.K = K
        self.xi = xi

        self.mean_reward_ = np.zeros(K)
        self.sum_pulling_fraction = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.pulling_list = [kk for kk in range(1, K + 1)]

        C = 10  # I am not sure C=10 is enough to fulfill the requirement
        self.beta = lambda x: np.log(C) + 2 * np.log(x) + np.log(1 / delta)
        self.function_f = lambda x: np.log(C) + 10 * np.log(x)

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert not self.stop, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = (
            self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        )
        self.t += 1

        # calculate the arm to be pulled in the next round
        if len(self.pulling_list) == 0:
            # It = self.Get_It(self.mean_reward_, self.pulling_times_)
            # it = It[0]
            # wt = self.Get_wt(self.mean_reward_, it=it)
            wt = self.Get_wt(self.mean_reward_, self.pulling_times_)
            ## C-Track
            epsilon = 1 / np.sqrt(self.K**2 + self.t)
            projected_w = self.get_projection(wt, epsilon)
            self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w
            arm = np.argmax(self.sum_pulling_fraction - self.pulling_times_) + 1
            self.pulling_list.append(arm)

        # determine whether to stop
        max_mean = np.max(self.mean_reward_)
        if max_mean > self.xi:
            a0 = np.argmax(self.mean_reward_) + 1
            beta_t = self.beta(self.t - 1)
            condition = (
                self.pulling_times_[a0 - 1]
                * (self.mean_reward_[arm - 1] - self.xi) ** 2
                / 2
            )
            if beta_t < condition:
                self.stop = True
                return a0
        else:
            for arm in np.arange(1, self.K + 1):
                beta_t = self.beta(self.t - 1)
                condition = (
                    self.pulling_times_[arm - 1]
                    * (self.mean_reward_[arm - 1] - self.xi) ** 2
                    / 2
                )
                if condition <= beta_t:
                    # that means we can find an instance that is in both $\neg i$
                    # and $\mathcal{D}_t$
                    return None
            self.stop = True
            return "No Arms Above xi"

    def if_stop(self):
        return self.stop

    def get_projection(self, w, epsilon):
        # project the w into the $[\epsilon, 1]^K \cap \Sigma_K$, through solving linear optimization problem
        # Please check README.md to see why the following codes can find the projection
        projected_w = np.zeros(self.K)
        threshold_index = w < epsilon
        projected_w[threshold_index] = epsilon

        gap = np.sum(np.maximum(epsilon - w, 0))
        projected_w[~threshold_index] = w[~threshold_index] - gap / (
            np.sum(~threshold_index)
        )

        return projected_w

    def Get_wt(self, hatmu, pulling):
        max_mean = np.max(hatmu)
        if max_mean < self.xi:
            wt = 2 / (hatmu - self.xi) ** 2
            wt = wt / np.sum(wt)
            return wt
        else:  # max_mean \geq self.xi
            it = self.Get_it(hatmu, pulling)
            if hatmu[it - 1] > self.xi:
                wt = np.zeros(self.K)
                wt[it - 1] = 1
                return wt
            else:
                wt = np.ones(self.K) / self.K
                return wt

    def Get_it(self, hatmu: np.ndarray, pulling: np.ndarray):
        best_emp_arm = np.argmax(hatmu) + 1
        best_emp = hatmu[best_emp_arm - 1]
        if best_emp < self.xi:
            # $\max_a \hat{\mu}_{a,t} < xi$
            return 1

        ft = self.function_f(self.t - 1)
        mu_temp = hatmu.copy()
        mu_temp[hatmu > self.xi] = self.xi
        condition = np.sum(pulling * (mu_temp - hatmu) ** 2 / 2)
        if condition < ft:
            return 1

        for arm in range(1, best_emp_arm):
            if hatmu[arm - 1] > self.xi:
                hat_mu_arm_t = hatmu[arm - 1]
                N_arm_t = pulling[arm - 1]
                index_arm = hatmu > hat_mu_arm_t
                mu_temp = hatmu[index_arm]
                Nt = pulling[index_arm]
                f = lambda x: np.sum(
                    N_arm_t * (x - hat_mu_arm_t) ** 2
                    + Nt * (np.maximum(mu_temp - x, 0) ** 2)
                )

                if gss(f, hatmu[arm - 1], best_emp, ft):
                    return arm
            else:
                hat_mu_arm_t = hatmu[arm - 1]
                N_arm_t = pulling[arm - 1]
                index_better_than_arm = hatmu > self.xi
                mu_temp = hatmu[index_better_than_arm]
                Nt = pulling[index_better_than_arm]

                f = lambda x: np.sum(
                    N_arm_t * (x - hat_mu_arm_t) ** 2
                    + Nt * (np.maximum(mu_temp - x, 0) ** 2)
                )

                if gss(f, self.xi, best_emp, ft):
                    return arm

        # if the algorithm doesn't return any value
        # then best_emp_arm is the minimum arm index that's in I_t
        return best_emp_arm


class Sticky_TaS_fast(object):
    # Sticky Track-and-Stop
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        self.delta = delta
        self.K = K
        self.xi = xi

        self.mean_reward_ = np.zeros(K)
        self.sum_pulling_fraction = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.pulling_list = [kk for kk in range(1, K + 1)]

        C = 10  # I am not sure C=10 is enough to fulfill the requirement
        self.beta = lambda x: np.log(C) + 2 * np.log(x) + np.log(1 / delta)
        self.function_f = lambda x: np.log(C) + 10 * np.log(x)

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert not self.stop, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = (
            self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        )
        self.t += 1

        # calculate the arm to be pulled in the next round
        if len(self.pulling_list) == 0:
            # It = self.Get_It(self.mean_reward_, self.pulling_times_)
            # it = It[0]
            # wt = self.Get_wt(self.mean_reward_, it=it)
            wt = self.Get_wt(self.mean_reward_, self.pulling_times_)
            ## C-Track
            epsilon = 1 / np.sqrt(self.K**2 + self.t)
            projected_w = self.get_projection(wt, epsilon)
            self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w
            arm = np.argmax(self.sum_pulling_fraction - self.pulling_times_) + 1
            self.pulling_list.append(arm)

        # determine whether to stop
        max_mean = np.max(self.mean_reward_)
        if max_mean > self.xi:
            a0 = np.argmax(self.mean_reward_) + 1
            beta_t = self.beta(self.t - 1)
            condition = (
                self.pulling_times_[a0 - 1]
                * (self.mean_reward_[arm - 1] - self.xi) ** 2
                / 2
            )
            if beta_t < condition:
                self.stop = True
                return a0
        else:
            for arm in np.arange(1, self.K + 1):
                beta_t = self.beta(self.t - 1)
                condition = (
                    self.pulling_times_[arm - 1]
                    * (self.mean_reward_[arm - 1] - self.xi) ** 2
                    / 2
                )
                if condition <= beta_t:
                    # that means we can find an instance that is in both $\neg i$
                    # and $\mathcal{D}_t$
                    return None
            self.stop = True
            return "No Arms Above xi"

    def if_stop(self):
        return self.stop

    def get_projection(self, w, epsilon):
        # project the w into the $[\epsilon, 1]^K \cap \Sigma_K$, through solving linear optimization problem
        # Please check README.md to see why the following codes can find the projection
        projected_w = np.zeros(self.K)
        threshold_index = w < epsilon
        projected_w[threshold_index] = epsilon

        gap = np.sum(np.maximum(epsilon - w, 0))
        projected_w[~threshold_index] = w[~threshold_index] - gap / (
            np.sum(~threshold_index)
        )

        return projected_w

    def Get_wt(self, hatmu, pulling):
        max_mean = np.max(hatmu)
        if max_mean < self.xi:
            wt = 2 / (hatmu - self.xi) ** 2
            wt = wt / np.sum(wt)
            return wt
        else:  # max_mean \geq self.xi
            it = self.Get_it(hatmu, pulling)
            if hatmu[it - 1] > self.xi:
                wt = np.zeros(self.K)
                wt[it - 1] = 1
                return wt
            else:
                wt = np.ones(self.K) / self.K
                return wt

    def Get_it(self, hatmu: np.ndarray, pulling: np.ndarray):
        best_emp_arm = np.argmax(hatmu) + 1
        best_emp = hatmu[best_emp_arm - 1]
        if best_emp < self.xi:
            # $\max_a \hat{\mu}_{a,t} < xi$
            return 1

        ft = self.function_f(self.t - 1)
        mu_temp = hatmu.copy()
        mu_temp[hatmu > self.xi] = self.xi
        condition = np.sum(pulling * (mu_temp - hatmu) ** 2 / 2)
        if condition < ft:
            return 1

        arms_above_xi = np.where(hatmu > self.xi) + 1  # O(K)
        mu_above_xi = hatmu[arms_above_xi - 1]  # O(K)

        not_included_arm_ = []
        for arm in range(1, best_emp_arm):
            hat_mu_arm_t = hatmu[arm - 1]
            N_arm_t = pulling[arm - 1]
            for bench_Nt, bench_mean in not_included_arm_:
                if N_arm_t >= bench_Nt and hat_mu_arm_t <= bench_mean:
                    continue

            if hatmu[arm - 1] > self.xi:
                index_arm = hatmu > hat_mu_arm_t
                mu_temp = hatmu[index_arm]
                Nt = pulling[index_arm]
                f = lambda x: np.sum(
                    N_arm_t * (x - hat_mu_arm_t) ** 2
                    + Nt * (np.maximum(mu_temp - x, 0) ** 2)
                )

                if gss(f, hatmu[arm - 1], best_emp, ft):
                    return arm
                else:
                    not_included_arm_ = [
                        (nt, hatmu)
                        for (nt, hatmu) in not_included_arm_
                        if nt < N_arm_t or hatmu >= hat_mu_arm_t
                    ]
                    not_included_arm_.append((hat_mu_arm_t, N_arm_t))
            else:
                continue

        # if the algorithm doesn't return any value
        # then best_emp_arm is the minimum arm index that's in I_t
        return best_emp_arm


# %% unit test 1, test Sticky_TaS
# from env import Environment_Gaussian

# rlist = [0.1, 0.2, 0.0, 0.6]
# K = len(rlist)
# xi = 0.5
# delta = 0.001

# env = Environment_Gaussian(rlist=rlist, K=K, random_seed=12345)
# agent = Sticky_TaS_old(K=K, delta=delta, xi=xi)
# # agent = Sticky_TaS(K=K, delta=delta, xi=xi)
# output_arm = None
# stop_time = 0
# while not agent.stop:
#     arm = agent.action()
#     reward = env.response(arm)
#     output_arm = agent.observe(reward)
#     if output_arm is not None:
#         predicted_arm = output_arm
#         stop_time = agent.t
# print(f"output arm is {output_arm}, output time is {stop_time}")

# %% unit test 2, compare the running speed of Sticky_TaS and Sticky_TaS_old
from env import Environment_Gaussian
from tqdm import tqdm
from time import time

K = 50
xi = 0.5
Delta = 0.01
rlist = np.ones(K) * xi
rlist[1 : (K + 1) // 2] = xi + Delta
rlist[(K + 1) // 2 : K] = xi - Delta
rlist[0] = 1.0

delta = 0.01
n_exp = 100


# for alg_class in [Sticky_TaS_old, Sticky_TaS]:
for alg_class in [Sticky_TaS_fast, Sticky_TaS]:
    stop_time_ = np.zeros(n_exp)
    output_arm_ = list()
    correctness_ = np.ones(n_exp)
    exectution_time_ = np.zeros(n_exp)
    for exp_id in tqdm(range(n_exp)):
        rlist_temp = rlist.copy()
        np.random.seed(exp_id)
        np.random.shuffle(rlist_temp)
        answer_set = list(np.where(rlist_temp > xi)[0] + 1)

        env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=exp_id)
        agent = alg_class(K=K, delta=delta, xi=xi)

        time_start = time()
        while not agent.stop:
            arm = agent.action()
            reward = env.response(arm)
            output_arm = agent.observe(reward)
            if output_arm is not None:
                output_arm_.append(output_arm)
                break
        time_end = time()
        stop_time_[exp_id] = agent.t
        exectution_time_[exp_id] = time_end - time_start
        if output_arm not in answer_set:
            correctness_[exp_id] = 0
    mean_stop_time = np.mean(stop_time_)
    mean_success = np.mean(correctness_)
    mean_execution_time = np.mean(exectution_time_)
    algname = type(agent).__name__
    print(f"For algorithm {algname}, ")
    print(f"mean stop time is {mean_stop_time}")
    print(f"correctness rate is {mean_success}")
    print(f"execution time is {mean_execution_time}")
# """ output
# K = 100, xi = 0.5, Delta = 0.01, delta = 0.01, n_exp = 100
# rlist = np.ones(K) * xi
# rlist[1 : (K + 1) // 2] = xi + Delta
# rlist[(K + 1) // 2 : K] = xi - Delta
# rlist[0] = 1.0

# For algorithm Sticky_TaS_old,
# mean stop time is 255423.87
# correctness rate is 1.0
# execution time is 24.250074293613434

# For algorithm Sticky_TaS,
# mean stop time is 255423.87
# correctness rate is 1.0
# execution time is 22.604291009902955
# """

# %% unit test 3, how would the permutaion of arms affect the pulling complexity?
# the gap can be 10 times larger
# from env import Environment_Gaussian
# from tqdm import tqdm
# from time import time

# K = 100
# xi = 0.5
# Delta = 0.01
# rlist = np.ones(K) * xi
# rlist[1 : (K + 1) // 2] = xi + Delta
# rlist[(K + 1) // 2 : K] = xi - Delta
# rlist[0] = 1.0
# delta = 0.01
# n_exp = 100


# for alg_class in [Sticky_TaS]:
#     stop_time_ = np.zeros(n_exp)
#     output_arm_ = list()
#     correctness_ = np.ones(n_exp)
#     exectution_time_ = np.zeros(n_exp)
#     for exp_id in tqdm(range(n_exp)):
#         rlist_temp = rlist.copy()
#         answer_set = list(np.where(rlist_temp > xi)[0] + 1)

#         env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=exp_id)
#         agent = alg_class(K=K, delta=delta, xi=xi)

#         time_start = time()
#         while not agent.stop:
#             arm = agent.action()
#             reward = env.response(arm)
#             output_arm = agent.observe(reward)
#             if output_arm is not None:
#                 output_arm_.append(output_arm)
#                 break
#         time_end = time()
#         stop_time_[exp_id] = agent.t
#         exectution_time_[exp_id] = time_end - time_start
#         if output_arm not in answer_set:
#             correctness_[exp_id] = 0
#     mean_stop_time = np.mean(stop_time_)
#     mean_success = np.mean(correctness_)
#     mean_execution_time = np.mean(exectution_time_)
#     algname = type(agent).__name__
#     print(f"For algorithm {algname}, ")
#     print(f"mean stop time is {mean_stop_time}")
#     print(f"correctness rate is {mean_success}")
#     print(f"execution time is {mean_execution_time}")

# # reverse the order of rlist
# for alg_class in [Sticky_TaS]:
#     stop_time_ = np.zeros(n_exp)
#     output_arm_ = list()
#     correctness_ = np.ones(n_exp)
#     exectution_time_ = np.zeros(n_exp)
#     for exp_id in tqdm(range(n_exp)):
#         rlist_temp = rlist[::-1].copy()
#         answer_set = list(np.where(rlist_temp > xi)[0] + 1)

#         env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=exp_id)
#         agent = alg_class(K=K, delta=delta, xi=xi)

#         time_start = time()
#         while not agent.stop:
#             arm = agent.action()
#             reward = env.response(arm)
#             output_arm = agent.observe(reward)
#             if output_arm is not None:
#                 output_arm_.append(output_arm)
#                 break
#         time_end = time()
#         stop_time_[exp_id] = agent.t
#         exectution_time_[exp_id] = time_end - time_start
#         if output_arm not in answer_set:
#             correctness_[exp_id] = 0
#     mean_stop_time = np.mean(stop_time_)
#     mean_success = np.mean(correctness_)
#     mean_execution_time = np.mean(exectution_time_)
#     algname = type(agent).__name__
#     print(f"For algorithm {algname}, ")
#     print(f"mean stop time is {mean_stop_time}")
#     print(f"correctness rate is {mean_success}")
#     print(f"execution time is {mean_execution_time}")
