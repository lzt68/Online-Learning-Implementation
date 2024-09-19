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
