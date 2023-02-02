# README

We aim to reproduce the figure 2 sparse case in Jun & Kowak 2016

## File Structure

"Numeric-Record": Folder for Numeric Record

"Source": Source file of the algorithm and environment

+ agent.py
  + Uniform_Agent: adopt round robin to determine the pulling arm in each round
  + UCB_Agent: adopt upper confidence bound to determine the pulling arm in each round, details can be found Bubeck et.al 2009 "Pure Exploration in Multi-armed Bandits Problems" 
  + SequentialHalving_FixedBudget_Agent: sequnetial halving algorithm in Karnin et.al 2013 "Almost Optimal Exploration in Multi-Armed Bandits"
  + DoublingSequentialHalving_ATMean_Agent: Apply doubling trick on SequentialHalving_FixedBudget_Agent to make it an anytime algorithm. But here we follow Jun & Nowak 2016, using the empirical mean reward to predict the best arm in each round
  + AT_LUCB_Agent: The anytime lower and upper confidence bound algorithm proposed by Jun & Nowak 2016
+ env.py
  + Env_FixedConsumption
    the reward is stochastic, the consumption is fixed
  + Env_Uncorrelated_Reward
    In this environment, the reward and demand are independent. Both following Bernoulli Distribution.
  + Env_Correlated_Uniform
    In this environment, the reward and demand are dependent
    $reward = \mathbb{1}(U <= r)$, $consumption = \mathbb{1}(U <= d)$,
    where $U$ follows uniform distribution on $(0, 1)$
  + class Env_FixedConsumption_Gaussian
    In this environment, the reward follows gaussian distribution with variance 1/4,
    The consumption is fixed, all equal to 1

Experiment.ipynb: Reproduce Figure 2 Sparse Case in the paper. But I met some problems regarding the curve of doubling-trick-sequential-halving

