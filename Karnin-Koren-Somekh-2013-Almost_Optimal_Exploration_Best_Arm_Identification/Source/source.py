from __future__ import annotations
import numpy as np
from tqdm import tqdm


class Env_FixedConsumption:
    def __init__(self, r_list=[0.5, 0.25], d_list=[0.1, 0.1], K=2, C=10, random_seed=12345) -> None:
        """In this environment, the reward is stochastic, the consumption is fixed
        Args:
            r_list (list, optional): The mean reward of each arm. Defaults to [0.5, 0.25].
            d_list (list, optional): The mean consumption of each arm. Defaults to [0.1, 0.1].
            K (int, optional): The total number of arms. Defaults to 2.
            C (int, optional): Initial Resource. Defaults to 10.
            random_seed (int, optional): Random seed. Defaults to 12345.
        """
        assert len(r_list) == len(d_list), "number of arms doesn't match"
        assert len(r_list) == K, "number of arms doesn't match"
        assert C > 0, "initial resource should be greater than 0"

        self.r_list = r_list
        self.d_list = d_list
        self.K = K
        self.C = C
        self.consumption = 0
        self.stop = False  # when the consumption > C-1, the algorithm stops
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def response(self, arm):
        if not self.stop:
            consumption = self.d_list[arm - 1]
            reward = np.random.binomial(n=1, p=self.r_list[arm - 1])
            self.consumption += consumption
            if self.consumption >= self.C - 1:
                self.stop = True
            return consumption, reward
        else:
            return None

    def if_stop(self):
        return self.stop


class SequentialHalving_Agent:
    # use round robin to pull remaining arms, and eliminate half of the remaining arms
    def __init__(self, K=2, C=10) -> None:
        """Construct an instance of Sequential Halving policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
        """
        self.K = K
        self.C = C
        self.t = 0  # index of round
        self.t_q = 0  # index of round in each phase
        self.q = 0  # index of phase
        self.arm_ = list()  # record the action in each epoch

        self.demand_ = dict()  # record the consumption of arms in each epoch
        self.reward_ = dict()  # record the observed reward of arms in each epoch
        self.consumption = 0  # record the consumption in each phase
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # when we enter a new epoch, we clear the existing memory
            self.demand_[arm_index] = list()
            self.reward_[arm_index] = list()
        self.pulling_times_ = np.zeros(K)

        self.total_demand_ = dict()
        self.total_reward_ = dict()
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # but we will not clear the memory
            self.total_demand_[arm_index] = list()
            self.total_reward_[arm_index] = list()
        self.total_consumption = 0  # the total consumption of all the phase

        self.survive_arms = list(range(1, K + 1))

    def action(self):
        # return the pulling arm in this epoch
        index = self.t_q % len(self.survive_arms)
        arm = self.survive_arms[index]
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this phase
        self.reward_[self.arm_[-1]].append(reward)
        self.demand_[self.arm_[-1]].append(demand)
        self.pulling_times_[self.arm_[-1] - 1] += 1

        # record the arms in this overall array
        self.total_reward_[self.arm_[-1]].append(reward)
        self.total_demand_[self.arm_[-1]].append(demand)

        # update the consumption
        self.consumption = self.consumption + demand
        self.total_consumption = self.total_consumption + demand

        # update the index of rounds
        self.t = self.t + 1
        self.t_q = self.t_q + 1

        if len(self.survive_arms) == 1:
            return

        # check whether conduct the elimination
        if self.consumption >= self.C / np.ceil(np.log2(self.K)) - 1:
            # we need to make sure all the arm share the same pulling times
            pulling_times = self.t_q // len(self.survive_arms)
            # mean_reward = self.mean_reward_[self.survive_arms]
            mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in self.survive_arms])
            # sort the mean reward with descending order
            sort_order = np.argsort(mean_reward)[::-1]
            self.survive_arms = np.array([self.survive_arms[ii] for ii in sort_order[: int(np.ceil(len(self.survive_arms) / 2))]])
            self.survive_arms = np.sort(self.survive_arms)

            self.t_q = 0
            for arm_index in self.survive_arms:
                self.demand_[arm_index] = list()
                self.reward_[arm_index] = list()
                self.consumption = 0
                self.q = self.q + 1

    def predict(self):
        # output the predicted best arm
        assert len(self.survive_arms) == 1
        best_arm = self.survive_arms[0]
        return best_arm


class SequentialHalving_Agent_Variant:
    # pre-caculate the pulling times of each arm in each phase
    def __init__(self, K=2, C=10) -> None:
        """Construct an instance of Sequential Halving policy

        Args:
            K (int, optional): Total number of arms. Defaults to 2.
            C (int, optional): Available initial resource. Defaults to 10.
        """
        self.K = K
        self.C = C
        self.t = 0  # index of round
        self.t_q = 0  # index of round in each phase
        self.q = 0  # index of phase
        self.arm_ = list()  # record the action in each epoch

        self.demand_ = dict()  # record the consumption of arms in each epoch
        self.reward_ = dict()  # record the observed reward of arms in each epoch
        self.consumption = 0  # record the consumption in each phase
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # when we enter a new epoch, we clear the existing memory
            self.demand_[arm_index] = list()
            self.reward_[arm_index] = list()
        self.pulling_times_ = np.zeros(K)

        self.total_demand_ = dict()
        self.total_reward_ = dict()
        for arm_index in range(1, K + 1):
            # for each arm, create a list
            # but we will not clear the memory
            self.total_demand_[arm_index] = list()
            self.total_reward_[arm_index] = list()
        self.total_consumption = 0  # the total consumption of all the phase

        self.survive_arms = list(range(1, K + 1))

        self.pulling_list = []
        for kk in range(1, K + 1):
            self.pulling_list = self.pulling_list + [kk] * int(np.floor(self.C / np.ceil(np.log2(self.K)) / self.K))

    def action(self):
        # return the pulling arm in this epoch
        assert len(self.pulling_list) > 0
        arm = self.pulling_list[0]
        self.pulling_list.pop(0)
        self.arm_.append(arm)
        return arm

    def observe(self, demand, reward):
        # record the arms in this phase
        self.reward_[self.arm_[-1]].append(reward)
        self.demand_[self.arm_[-1]].append(demand)
        self.pulling_times_[self.arm_[-1] - 1] += 1

        # record the arms in this overall array
        self.total_reward_[self.arm_[-1]].append(reward)
        self.total_demand_[self.arm_[-1]].append(demand)

        # update the consumption
        self.consumption = self.consumption + demand
        self.total_consumption = self.total_consumption + demand

        # update the index of rounds
        self.t = self.t + 1
        self.t_q = self.t_q + 1

        if len(self.survive_arms) == 1:
            return

        # check whether conduct the elimination
        if len(self.pulling_list) == 0:
            # we need to make sure all the arm share the same pulling times
            pulling_times = self.t_q // len(self.survive_arms)
            # mean_reward = self.mean_reward_[self.survive_arms]
            mean_reward = np.array([np.mean(self.reward_[ii][:pulling_times]) for ii in self.survive_arms])
            # sort the mean reward with descending order
            sort_order = np.argsort(mean_reward)[::-1]
            self.survive_arms = np.array([self.survive_arms[ii] for ii in sort_order[: int(np.ceil(len(self.survive_arms) / 2))]])
            self.survive_arms = np.sort(self.survive_arms)

            # generate pulling list
            self.pulling_list = []
            for arm in self.survive_arms:
                self.pulling_list = self.pulling_list + [arm] * int(np.floor(self.C / np.ceil(np.log2(self.K)) / len(self.survive_arms)))

            # clear the memory
            # self.t_q = 0
            # for arm_index in range(1, self.K + 1):
            #     self.demand_[arm_index] = list()
            #     self.reward_[arm_index] = list()
            #     self.consumption = 0
            #     self.q = self.q + 1
            self.t_q = 0
            for arm_index in self.survive_arms:
                self.demand_[arm_index] = list()
                self.reward_[arm_index] = list()
                self.consumption = 0
                self.q = self.q + 1

    def predict(self):
        # output the predicted best arm
        assert len(self.survive_arms) == 1
        best_arm = self.survive_arms[0]
        return best_arm


def OneSubOptimal_H_H(K: int, dstart: float = 0.2, dstop: float = 0.8):
    """Arms with higher reward consume less resource

    Args:
        K (int): Number of Arms
        dstart (float, optional): Smallest mean consumption. Defaults to 0.2.
        dstop (float, optional): Biggest mean consumption. Defaults to 0.8.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ndarray for reward, demand and price with length K
    """
    reward = np.ones(K) * 0.45
    reward[0] = 0.5
    # demand = np.linspace(start=dstart, stop=dstop, num=K)[::-1]  # reverse the order
    demand = np.linspace(start=dstart, stop=dstop, num=K)
    price = reward / demand
    return reward, demand, price


def Get_H1(reward: np.ndarray, demand: np.ndarray) -> float:
    # assume r_1>r_2>\cdots>r_K, $\{d_{(k)}\}_{k=1}^{K}$ is a permutation of $\{d_k\}_{k=1}^{K}$
    # $d_{(1)}\ge d_{(2)}\ge \cdots \ge d_{(K)}$
    # the classic defintion of $H_1=\frac{d_{(1)}}{(r_{(1)}-r_{(2)})^2}+\sum_{k=1}^K \frac{d_{(k)}}{(r_{(1)}-r_{(k)})^2}$
    reward = np.sort(reward)[::-1]
    # demand = np.sort(demand)[::-1]
    demand = np.ones(len(reward))
    gap = reward[0] - reward
    gap[0] = reward[0] - reward[1]
    H1 = np.sum(demand / gap**2)
    return H1


# temporate experiment oracle
def Experiment(
    reward: np.ndarray, demand: np.ndarray, env_class, env_para: dict, agent_class, agent_para: dict, n_experiment: int, K: int, C: int, random_seed: int = 0
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Experiment oracle

    Args:
        reward (np.ndarray): the mean reward of each arm
        demand (np.ndarray): the mean consumption of each arm
        env_class (_type_): the class of environment
        env_para (dict): extra parameter settings for the environment
        agent_class (_type_): the class of agent
        agent_para (dict): extra parameter settings for the agent
        n_experiment (int): execution times of independent experiment
        K (int) : arm numbers
        C (int) : initial available resource
        random_seed (int, optional) : random seed. default value is 0

    Returns:
        success_rate (float): the success rate in the experiments
        std_success_rate (float): the standard deviation of the success rate
            std_success_rate = sqrt(success_rate * (1 - success_rate)) / sqrt(n_experiment)
        stopping_times_ (np.ndarray): the ending round index
        predict_arm_ (np.ndarray): the predicted arm in each independent experiment copy
        best_arm_ (np.ndarray): the actual best arm in each independent experiment copy
    """
    best_arm_ = np.zeros(n_experiment)
    predict_arm_ = np.zeros(n_experiment)
    stopping_times_ = np.zeros(n_experiment)
    for exp_index in tqdm(range(n_experiment)):
        # different random_seed will generate different permutation
        np.random.seed(random_seed + exp_index)

        # permute the arm
        permuted_index = np.arange(K)
        # np.random.shuffle(permuted_index)
        reward = reward[permuted_index]
        demand = demand[permuted_index]
        best_arm_[exp_index] = np.argmax(reward) + 1

        # set up the parameters of environments and agents, and define the env and agent
        env_para["r_list"] = reward
        env_para["d_list"] = demand
        env_para["K"] = K
        env_para["C"] = C
        env_para["random_seed"] = random_seed + exp_index
        env = env_class(**env_para)

        agent_para["K"] = K
        agent_para["C"] = C
        agent = agent_class(**agent_para)

        # run the experiment
        while not env.if_stop():
            arm = agent.action()
            d, r = env.response(arm)
            agent.observe(demand=d, reward=r)
        predict_arm_[exp_index] = agent.predict()
        stopping_times_[exp_index] = agent.t

    # calculate the return value
    success_rate = np.mean(predict_arm_ == best_arm_)
    std_success_rate = np.sqrt(success_rate * (1 - success_rate)) / np.sqrt(n_experiment)
    stopping_times_ = np.mean(stopping_times_)

    return success_rate, std_success_rate, stopping_times_, predict_arm_, best_arm_
