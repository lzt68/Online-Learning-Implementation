# def Get_wt_old(self, hatmu, it):
#     max_mean = np.max(hatmu)
#     if max_mean < self.xi:
#         wt = 2 / (hatmu - self.xi) ** 2
#         wt = wt / np.sum(wt)
#         return wt
#     else:
#         # as we use sticky rule, $i_t\in i_F(\hat{\mu})$ might not hold
#         if hatmu[it - 1] > self.xi:
#             wt = np.zeros(self.K)
#             wt[it - 1] = 1
#             return wt
#         else:
#             # if $i_t\notin i_F(\hat{\mu})$,
#             # then $D(\vec{\mu}, \neg i)=0$, and we take any pulling fraction w
#             # here we take uniformly pulling fraction
#             wt = np.ones(self.K) / self.K
#             return wt

# def Get_It(self, hatmu: np.ndarray, pulling_times: np.ndarray):
#     max_mean = np.max(hatmu)
#     if max_mean < self.xi:
#         It = np.arange(1, self.K + 1)
#         return It
#     else:
#         It = []
#         arm_above_xi = list(np.where(hatmu >= self.xi)[0] + 1)
#         arm_below_xi = list(np.where(hatmu < self.xi)[0] + 1)
#         for arm in arm_above_xi:
#             if hatmu[arm - 1] == max_mean:
#                 It.append(arm)
#                 continue
#             else:
#                 mu_temp = hatmu.copy()
#                 mu_temp[hatmu > hatmu[arm - 1]] = hatmu[arm - 1]
#                 condition = np.sum(pulling_times * (mu_temp - hatmu) ** 2 / 2)
#                 ft = self.function_f(self.t - 1)
#                 if condition < ft:
#                     It.append(arm)

#         for arm in arm_below_xi:
#             mu_temp = hatmu.copy()
#             mu_temp[hatmu > self.xi] = self.xi
#             mu_temp[arm - 1] = self.xi
#             condition = np.sum(pulling_times * (mu_temp - hatmu) ** 2 / 2)
#             ft = self.function_f(self.t - 1)
#             if condition < ft:
#                 It.append(arm)
#         It.sort()
#         return It


# class Sticky_TaS_old(object):
#     # Sticky Track-and-Stop
#     def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
#         self.delta = delta
#         self.K = K
#         self.xi = xi

#         self.mean_reward_ = np.zeros(K)
#         self.sum_pulling_fraction = np.zeros(K)
#         self.pulling_times_ = np.zeros(K)
#         self.total_reward_ = np.zeros(K)
#         self.action_ = list()
#         self.t = 1

#         self.pulling_list = [kk for kk in range(1, K + 1)]

#         C = 10  # I am not sure C=10 is enough to fulfill the requirement
#         self.beta = lambda x: np.log(C) + 2 * np.log(x) + np.log(1 / delta)
#         self.function_f = lambda x: np.log(C) + 10 * np.log(x)

#         self.stop = False

#     def action(self):
#         assert not self.stop, "the algorithm stops"
#         assert len(self.pulling_list) > 0, "pulling list is empty"

#         arm = self.pulling_list.pop(0)
#         self.action_.append(arm)
#         return arm

#     def observe(self, reward):
#         assert not self.stop, "the algorithm stops"
#         arm = self.action_[self.t - 1]
#         self.total_reward_[arm - 1] += reward
#         self.pulling_times_[arm - 1] += 1
#         self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
#         self.t += 1

#         # calculate the arm to be pulled in the next round
#         if len(self.pulling_list) == 0:
#             It = self.Get_It(self.mean_reward_, self.pulling_times_)
#             it = It[0]
#             wt = self.Get_wt(self.mean_reward_, it=it)
#             ## C-Track
#             epsilon = 1 / np.sqrt(self.K**2 + self.t)
#             projected_w = self.get_projection(wt, epsilon)
#             self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w
#             arm = np.argmax(self.sum_pulling_fraction - self.pulling_times_) + 1
#             self.pulling_list.append(arm)

#         # determine whether to stop
#         max_mean = np.max(self.mean_reward_)
#         if max_mean > self.xi:
#             a0 = np.argmax(self.mean_reward_) + 1
#             beta_t = self.beta(self.t - 1)
#             condition = self.pulling_times_[a0 - 1] * (self.mean_reward_[arm - 1] - self.xi) ** 2 / 2
#             if beta_t < condition:
#                 self.stop = True
#                 return a0
#         else:
#             for arm in np.arange(1, self.K + 1):
#                 beta_t = self.beta(self.t - 1)
#                 condition = self.pulling_times_[arm - 1] * (self.mean_reward_[arm - 1] - self.xi) ** 2 / 2
#                 if condition <= beta_t:
#                     # that means we can find an instance that is in both $\neg i$
#                     # and $\mathcal{D}_t$
#                     return None
#             self.stop = True
#             return "No Arms Above xi"

#     def if_stop(self):
#         return self.stop

#     def get_projection(self, w, epsilon):
#         # project the w into the $[\epsilon, 1]^K \cap \Sigma_K$, through solving linear optimization problem
#         # Please check README.md to see why the following codes can find the projection
#         projected_w = np.zeros(self.K)
#         threshold_index = w < epsilon
#         projected_w[threshold_index] = epsilon

#         gap = np.sum(np.maximum(epsilon - w, 0))
#         projected_w[~threshold_index] = w[~threshold_index] - gap / (np.sum(~threshold_index))

#         return projected_w

#     def Get_wt(self, hatmu, it):
#         max_mean = np.max(hatmu)
#         if max_mean < self.xi:
#             wt = 2 / (hatmu - self.xi) ** 2
#             wt = wt / np.sum(wt)
#             return wt
#         else:
#             # as we use sticky rule, $i_t\in i_F(\hat{\mu})$ might not hold
#             if hatmu[it - 1] > self.xi:
#                 wt = np.zeros(self.K)
#                 wt[it - 1] = 1
#                 return wt
#             else:
#                 # if $i_t\notin i_F(\hat{\mu})$,
#                 # then $D(\vec{\mu}, \neg i)=0$, and we take any pulling fraction w
#                 # here we take uniformly pulling fraction
#                 wt = np.ones(self.K) / self.K
#                 return wt

#     def Get_It(self, hatmu: np.ndarray, pulling_times: np.ndarray):
#         max_mean = np.max(hatmu)
#         if max_mean < self.xi:
#             It = np.arange(1, self.K + 1)
#             return It
#         else:
#             It = []
#             arm_above_xi = list(np.where(hatmu >= self.xi)[0] + 1)
#             arm_below_xi = list(np.where(hatmu < self.xi)[0] + 1)
#             for arm in arm_above_xi:
#                 if hatmu[arm - 1] == max_mean:
#                     It.append(arm)
#                     continue
#                 else:
#                     mu_temp = hatmu.copy()
#                     mu_temp[hatmu > hatmu[arm - 1]] = hatmu[arm - 1]
#                     condition = np.sum(pulling_times * (mu_temp - hatmu) ** 2 / 2)
#                     ft = self.function_f(self.t - 1)
#                     if condition < ft:
#                         It.append(arm)

#             for arm in arm_below_xi:
#                 mu_temp = hatmu.copy()
#                 mu_temp[hatmu > self.xi] = self.xi
#                 mu_temp[arm - 1] = self.xi
#                 condition = np.sum(pulling_times * (mu_temp - hatmu) ** 2 / 2)
#                 ft = self.function_f(self.t - 1)
#                 if condition < ft:
#                     It.append(arm)
#             It.sort()
#             return It
