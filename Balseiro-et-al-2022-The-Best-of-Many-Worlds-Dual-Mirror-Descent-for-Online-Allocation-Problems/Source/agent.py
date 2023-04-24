import numpy as np
from scipy.optimize import linprog
from numpy.random import Generator, PCG64
from typing import Union


class H1_DualMirrorDescentOGD(object):
    def __init__(self, m: int, T: int, d: int, b: np.array, rho: Union[float, np.float64], eta: Union[float, np.float64], solver="linprog") -> None:
        """Implement the algorithm for the problem instance in section H1
        Here we adopt the reference function as $h(\mu)=\frac{1}{2}\|\mu\|_2^2$
        Thus the update fomula is $\mu_{t+1}=Proj_{\mu\ge 0}\{\mu_t-\eta g_t\}$

        Args:
            m (int): Number of resources.
            T (int): Number of rounds.
            d (int): Dimensions of decision variables.
            b (np.array): The initial available resource.
            rho (Union[float, np.float64]): Average budget of resources in each round.
            eta (Union[float, np.float64]): Update rates of dual variables.
            solver (str, optional): The setting of solving linear programme. Defaults to "linprog". Another choice
                is "SCIP".
        """
        assert b.shape[0] == m and len(b.shape) == 1, "Number of resources doesn't match"
        self.m = m
        self.T = T
        self.d = d
        self.b = b
        self.rho = rho
        self.eta = eta
        self.solver = solver
        self.remain_b = self.b.copy()

        self.r_t = np.zeros((d, T))
        self.reward = np.zeros(T)  # the generated reward in each round
        self.c_t = np.zeros((m, d, T))

        self.action_ = np.zeros((d, T))
        self.reward_ = np.zeros(T)

        self.mu_ = np.zeros((m, T))  # dual variable
        self.mu = np.zeros(m)

        self.t = 1
        self.k = 0

    def action(self, r_t, c_t):
        self.r_t[:, self.t - 1] = r_t
        self.c_t[:, :, self.t - 1] = c_t

        # calculate the decision variable
        tildex_t = self.action_get_tildex(r_t, c_t)
        if np.any(self.remain_b < c_t @ tildex_t):
            x_t = np.zeros(self.d)
        else:
            x_t = tildex_t.copy()
            self.remain_b -= c_t @ tildex_t
        self.action_[:, self.t - 1] = x_t
        self.reward[self.t - 1] = r_t @ x_t

        # update the dual variables
        self.mu = self.action_get_mu_tp1(c_t, tildex_t)
        self.mu_[:, self.t - 1] = self.mu.copy()

        self.t += 1
        return x_t

    def action_get_tildex(self, r_t, c_t):
        # calculate the decision variable
        if self.solver == "linprog":
            c = r_t - self.mu @ c_t

            A_eq = np.ones((1, self.d))
            b_eq = np.array([1.0])

            res = linprog(c=-c, A_eq=A_eq, b_eq=b_eq)
            action = res.x
        elif self.solver == "SCIP":
            import pyscipopt
            from pyscipopt import quicksum

            model = pyscipopt.Model()
            model.hideOutput()

            # define variable
            var = {}
            for xindex in range(1, self.d + 1):
                var[f"x{xindex}"] = model.addVar(vtype="C", lb=0.0, name=f"x{xindex}")

            # define object function
            c = r_t - self.mu @ c_t
            model.setObjective(
                quicksum(var[f"x{xindex}"] * c[xindex - 1] for xindex in range(1, self.d + 1)),
                "minimize",
            )

            # add constraint
            model.addCons(quicksum(var[f"x{xindex}"] for xindex in range(1, self.d + 1)) == 1)

            # optimize the model
            model.optimize()

            # update the p
            action = np.zeros(self.d)
            for xindex in range(1, self.d + 1):
                action[xindex - 1] = model.getVal(var[f"x{xindex}"])
        else:
            assert False, "Fail to find solver"

        return action

    def action_get_mu_tp1(self, c_t, action):
        g_t = -c_t @ action + self.rho
        mu = self.mu - self.eta * g_t
        mu[mu < 0.0] = 0.0
        return mu


#%% unit test 1, debug H1_OnlineLinearEnv and H1_DualMirrorDescentOGD
# from env import H1_OnlineLinearEnv
# import matplotlib.pyplot as plt

# m = 4
# T = 10
# d = 4
# random_seed = 0
# s = 1.0

# env = H1_OnlineLinearEnv(m=m, T=T, d=d, random_seed=random_seed)
# agent = H1_DualMirrorDescentOGD(m=m, T=T, d=d, b=env.b, rho=env.rho, eta=s / np.sqrt(T * m))
# while not env.if_stop():
#     r_t, c_t = env.deal()
#     action = agent.action(r_t=r_t, c_t=c_t)
#     env.observe(action)

# # calculate the upper bound
# mu_mean = np.cumsum(agent.mu_, axis=1) / np.arange(1, T + 1)
# reward_upper_bound = np.zeros(T)
# for tt in range(0, T):
#     for tt_ in range(0, tt + 1):
#         reward_upper_bound[tt] += np.max(agent.r_t[:, tt_] - mu_mean[:, tt] @ agent.c_t[:, :, tt_])
#     reward_upper_bound[tt] += (tt + 1) * agent.rho @ mu_mean[:, tt]

# reward_upper_bound_temp = np.zeros(T)
# for tt in range(0, T):
#     reward_upper_bound_temp[tt] = np.sum(np.max(agent.r_t[:, 0 : tt + 1] - np.einsum("i,ijk->jk", mu_mean[:, tt], agent.c_t[:, :, 0 : tt + 1]), axis=0))
#     reward_upper_bound_temp[tt] += (tt + 1) * agent.rho @ mu_mean[:, tt]
# print(np.abs(np.max(reward_upper_bound_temp - reward_upper_bound)))

# reward_agent = np.cumsum(agent.reward)

# plt.plot(reward_upper_bound - reward_agent)
# plt.show()
