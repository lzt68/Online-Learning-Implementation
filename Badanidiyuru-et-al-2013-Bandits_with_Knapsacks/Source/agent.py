from typing import Union
import numpy as np


class BalancedExploration(object):
    def __init__(self) -> None:
        pass

    def action(self):
        pass

    def observe(self):
        pass


class PrimalDualBwK(object):
    def __init__(self, d: int, m: int, B: Union[np.float64, int, float], Crad: Union[np.float64, float]) -> None:
        """Implement the Primal Dual algorithm

        Args:
            d (int): Number of resources
            m (int): Number of arms
            B (Union[np.float64, int, float]): Initial Budget. Here we assume the initial budget of
                all the resources are the same
            Crad (Union[np.float64, float]): The constant in calculating the radius of confidence interval
        """
        self.d = d
        self.m = m
        self.B = B
        self.Crad = Crad

        self.pulling_list = list(range(1, m + 1))
        self.v = np.ones(d)
        self.epsilon = np.sqrt(np.log(d) / B)

        self.t = 1
        self.action_ = list()
        self.total_reward_ = np.zeros(m)
        self.total_consumption_ = np.zeros((d, m))  # each column corresponds to an arm
        self.pulling_times_ = np.zeros(m)
        self.mean_reward_ = np.zeros(m)
        self.mean_consumption_ = np.zeros((d, m))

    def action(self):
        assert len(self.pulling_list) > 0, "fail to generate pulling arm"
        action = self.pulling_list.pop(0)
        self.action_.append(action)
        return action

    def observe(self, reward, consumption):
        assert len(consumption) == self.d, "The dimension of consumption doesn't match"

        # update the record
        arm = self.action_[-1]
        arm_index = arm - 1
        self.total_reward_[arm_index] += reward
        self.total_consumption_[:, arm_index] += consumption
        self.pulling_times_[arm_index] += 1
        self.mean_reward_[arm_index] = self.total_reward_[arm_index] / self.pulling_times_[arm_index]
        self.mean_consumption_[:, arm_index] = self.total_consumption_[:, arm_index] / self.pulling_times_[arm_index]

        if self.t <= self.m - 1:
            # In the initialization phase, we pull each arm once
            self.t += 1
            return

        # when t>=m+1, we need to confirm we the pulling list is empty before generating new pulling arm
        assert len(self.pulling_list) == 0, "The pulling list is not empty"

        # calculate the ucb and lcb
        ucb = self.mean_reward_ + self.rad(self.mean_reward_, self.pulling_times_)
        ucb = np.minimum(ucb, np.ones(self.m))
        ucb = np.maximum(ucb, np.zeros(self.m))
        lcb = self.mean_consumption_ + self.rad(self.mean_consumption_, np.tile(self.pulling_times_, (self.d, 1)))
        lcb = np.minimum(lcb, np.ones((self.d, self.m)))
        lcb = np.maximum(lcb, np.zeros((self.d, self.m)))
        EstCost = self.v @ lcb
        assert len(EstCost) == self.m, "The size of expected cost doesn't match with arm number"
        x = np.argmax(ucb / EstCost) + 1
        self.pulling_list.append(x)

        # update the v
        self.v = self.v * (1 + self.epsilon) ** lcb[:, x - 1]
        self.v = self.v / np.sum(np.abs(self.v))  # to avoid too big value of v, we devide v by its one norm

        self.t += 1

    def rad(self, v, N):
        radius = np.sqrt(self.Crad * v / N) + self.Crad / N
        return radius


#%% unit test 1, debug PrimalDualBwK and Env
from env import Env_FixedConsumption, Env_Uncorrelated_Reward, Env_Correlated_Uniform

random_seed = 12345
np.random.seed(random_seed)

d = 3
m = 5
B = 10
r_list = np.random.uniform(low=0.0, high=1.0, size=m)
d_list = np.random.uniform(low=0.0, high=1.0, size=(d, m))

print("Env_FixedConsumption")
env = Env_FixedConsumption(r_list=r_list, d_list=d_list, m=m, B=B, d=d, random_seed=random_seed)
agent = PrimalDualBwK(d=d, m=m, B=B, Crad=1.0)
reward_ = list()
while not env.if_stop():
    arm = agent.action()
    consumption, reward = env.response(arm=arm)
    agent.observe(reward=reward, consumption=consumption)
    reward_.append(reward)
print(f"Total reward is {np.sum(reward_)}")

print("Env_Uncorrelated_Reward")
env = Env_Uncorrelated_Reward(r_list=r_list, d_list=d_list, m=m, B=B, d=d, random_seed=random_seed)
agent = PrimalDualBwK(d=d, m=m, B=B, Crad=1.0)
reward_ = list()
while not env.if_stop():
    arm = agent.action()
    consumption, reward = env.response(arm=arm)
    agent.observe(reward=reward, consumption=consumption)
    reward_.append(reward)
print(f"Total reward is {np.sum(reward_)}")

print("Env_Correlated_Uniform")
env = Env_Correlated_Uniform(r_list=r_list, d_list=d_list, m=m, B=B, d=d, random_seed=random_seed)
agent = PrimalDualBwK(d=d, m=m, B=B, Crad=1.0)
reward_ = list()
while not env.if_stop():
    arm = agent.action()
    consumption, reward = env.response(arm=arm)
    agent.observe(reward=reward, consumption=consumption)
    reward_.append(reward)
print(f"Total reward is {np.sum(reward_)}")
