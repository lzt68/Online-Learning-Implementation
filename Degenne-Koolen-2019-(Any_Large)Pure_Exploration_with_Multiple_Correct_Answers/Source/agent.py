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

        # pre-calculate value for the sequential testing
        # arms_above_xi, num_arms_above_xi = self.Get_Arms_Above_xi(hatmu)
        arms_above_xi = self.Get_Arms_Above_xi(hatmu)
        sorted_mu_above_xi, sorted_pulling_above_xi, arm_order = (
            self.Get_Sorted_mu_pulling_slice(arms_above_xi, hatmu, pulling)
        )
        center_point, cum_sorted_pulling_above_xi = self.Get_center_point(
            sorted_mu_above_xi, sorted_pulling_above_xi
        )
        # cum_sorted_mu_above_xi, cum_sorted_pulling_mu_above_xi, cum2_sorted_pulling_mu_above_xi = self.Get_center_point(
        #     sorted_mu_above_xi, sorted_pulling_above_xi
        # )
        cum_sorted_pulling_mu_above_xi, cum2_sorted_pulling_mu_above_xi = (
            self.Get_center_point(sorted_mu_above_xi, sorted_pulling_above_xi)
        )
        for arm in range(1, self.K + 1):
            if hatmu[arm - 1] < self.xi:
                continue
            if arm_order[arm - 1] == 0:
                # print("new method: ", arm, hatmu[arm - 1], 0)
                return arm

            N_arm = pulling[arm - 1]
            mu_arm = hatmu[arm - 1]

            # use bisection to search for the optimal center point
            leftindex = 0
            rightindex = arm_order[arm - 1] - 1
            while True:
                middle_index = (leftindex + rightindex) // 2
                new_center_point = (
                    center_point[middle_index]
                    * cum_sorted_pulling_above_xi[middle_index]
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
                        * (
                            N_arm * mu_arm
                            + cum_sorted_pulling_mu_above_xi[middle_index]
                        )
                        * new_center_point
                    )
                    fval += (
                        N_arm * mu_arm**2
                        + cum2_sorted_pulling_mu_above_xi[middle_index]
                    )
                    fval /= 2
                    if fval < ft:
                        return arm
                        # print("new method:", arm, new_center_point, fval)
                        # find_or_not = True
                    break
                elif new_center_point < middle_next_mean:
                    leftindex = middle_index + 1
                elif new_center_point > sorted_mu_above_xi[middle_index]:
                    rightindex = middle_index

        assert False, "Failed to find $i_t$"

    def Get_Arms_Above_xi(self, hatmu):
        arms_above_xi = np.where(hatmu > self.xi)[0] + 1  # O(K)
        return arms_above_xi
        # num_arms_above_xi = len(arms_above_xi)
        # return arms_above_xi, num_arms_above_xi

    def Get_Sorted_mu_pulling_slice(self, arms_above_xi, hatmu, pulling):
        mu_above_xi = hatmu[arms_above_xi - 1]  # O(K)
        pulling_above_xi = pulling[arms_above_xi - 1]  # O(K)
        sorted_index_mu_above_xi = np.argsort(mu_above_xi)[::-1]  # O(K log K)
        # from largest to minimum

        sorted_mu_above_xi = mu_above_xi[sorted_index_mu_above_xi]  # O(K)
        sorted_pulling_above_xi = pulling_above_xi[sorted_index_mu_above_xi]  # O(K)
        # all the empirical mean above xi, sorted from largest to smallest
        # all the arm index whose empirical mean is above xi, same order as sorted_mu_above_xi

        mu_order_index = np.argsort(hatmu)[::-1]
        arm_order = np.argsort(mu_order_index)
        # arm_order[i-1] is the number of arms whose mu is greater than i^th arm

        return sorted_mu_above_xi, sorted_pulling_above_xi, arm_order

    def Get_center_point(self, sorted_mu_above_xi, sorted_pulling_above_xi):
        mu_times_Nt = sorted_mu_above_xi * sorted_pulling_above_xi
        cum_sorted_pulling_above_xi = np.cumsum(sorted_pulling_above_xi)
        total_largest_sum_mu_times_Nt = np.cumsum(mu_times_Nt)
        center_point = total_largest_sum_mu_times_Nt / cum_sorted_pulling_above_xi
        return center_point, cum_sorted_pulling_above_xi

    def Get_order_sum(self, sorted_mu_above_xi, sorted_pulling_above_xi):
        # cum_sorted_mu_above_xi = np.cumsum(sorted_mu_above_xi)  # first order sum
        cum_sorted_pulling_mu_above_xi = np.cumsum(
            sorted_pulling_above_xi * sorted_mu_above_xi
        )
        cum2_sorted_pulling_mu_above_xi = np.cumsum(
            sorted_pulling_above_xi * sorted_mu_above_xi**2
        )
        return cum_sorted_pulling_mu_above_xi, cum2_sorted_pulling_mu_above_xi
        # return cum_sorted_mu_above_xi, cum_sorted_pulling_mu_above_xi, cum2_sorted_pulling_mu_above_xi


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
# from env import Environment_Gaussian
# from tqdm import tqdm
# from time import time

# # K = 500
# # xi = 0.5
# # Delta = 0.01
# # rlist = np.ones(K) * xi
# # rlist[1 : (K + 1) // 2] = xi + Delta
# # rlist[(K + 1) // 2 : K] = xi - Delta
# # rlist[0] = 1.0

# K = 1000
# xi = 0.5
# Delta = 0.01
# rlist = np.ones(K) * xi
# rlist[1:K] = xi + Delta
# rlist[0] = 1.0

# delta = 0.01
# n_exp = 1

# # for alg_class in [Sticky_TaS_old, Sticky_TaS]:
# for alg_class in [Sticky_TaS_fast, Sticky_TaS]:
#     stop_time_ = np.zeros(n_exp)
#     output_arm_ = list()
#     correctness_ = np.ones(n_exp)
#     exectution_time_ = np.zeros(n_exp)
#     # for exp_id in tqdm(range(n_exp)):
#     for exp_id in range(n_exp):
#         rlist_temp = rlist[::-1].copy()
#         # np.random.seed(exp_id)
#         # np.random.shuffle(rlist_temp)
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
#             if agent.t % 10000 == 0:
#                 print(agent.t)
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

# %% unit test 4, check whether the actions are the same between Sticky_TaS_fast, Sticky_TaS
# from env import Environment_Gaussian
# from tqdm import tqdm
# from time import time
# import matplotlib.pyplot as plt

# K = 1000
# xi = 0.5
# Delta = 0.01
# rlist = np.ones(K) * xi
# rlist[1:K] = xi + Delta
# rlist[0] = 1.0

# delta = 0.01
# n_exp = 1


# rlist_temp = rlist[::-1].copy()
# # np.random.seed(exp_id)
# # np.random.shuffle(rlist_temp)
# answer_set = list(np.where(rlist_temp > xi)[0] + 1)

# env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=1)
# agent_sas = Sticky_TaS(K=K, delta=delta, xi=xi)
# agent_sas_fast = Sticky_TaS_fast(K=K, delta=delta, xi=xi)

# len_statistic = K * 10
# execution_time_fast = np.zeros(len_statistic)
# execution_time = np.zeros(len_statistic)

# count_round = 0
# while (not agent_sas.stop) or (not agent_sas_fast.stop):
#     # start_time_fast = time()
#     # arm_sas_fast = agent_sas_fast.action()
#     # end_time_fast = time()

#     # start_time = time()
#     # arm_sas = agent_sas.action()
#     # end_time = time()

#     # arm_it_fast = agent_sas_fast.Get_it(
#     #     agent_sas_fast.mean_reward_, agent_sas_fast.pulling_times_
#     # )
#     # arm_it = agent_sas.Get_it(agent_sas.mean_reward_, agent_sas.pulling_times_)

#     # assert (
#     #     arm_sas_fast == arm_sas and arm_it_fast == arm_it
#     # ), f"round {agent_sas.t} inconsistent"

#     arm_sas_fast = agent_sas_fast.action()
#     arm_sas = agent_sas.action()
#     assert arm_sas_fast == arm_sas, f"round {agent_sas.t} inconsistent"
#     assert agent_sas.stop == agent_sas_fast.stop, f"round {agent_sas.t} inconsistent"

#     reward = env.response(arm_sas_fast)

#     agent_sas.observe(reward)
#     agent_sas_fast.observe(reward)

#     # if agent_sas_fast.t <= len_statistic:
#     #     execution_time_fast[count_round] = end_time_fast - start_time_fast
#     #     execution_time[count_round] = end_time - start_time
#     #     count_round += 1
#     # else:
#     #     break

# print("fast:", np.mean(execution_time_fast))
# # print("old:", np.mean(execution_time))
# plt.figure(1)
# plt.plot(
#     range(2 * K, len_statistic),
#     execution_time_fast[2 * K : len_statistic],
#     label="fast",
# )
# plt.plot(
#     range(2 * K, len_statistic), execution_time[2 * K : len_statistic], label="normal"
# )
# plt.legend()
# plt.show()
