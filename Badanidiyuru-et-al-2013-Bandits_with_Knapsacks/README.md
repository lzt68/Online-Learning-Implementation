# README

This folder aims to implement the **PrimalDualBwK** in *Badanidiyuru et al. 2013 Bandits_with_Knapsacks*. Link of the paper: http://arxiv.org/abs/1305.2545. **I don't implement the BalancedExploration**. Though it is a well-defined mapping from the history to the action, it is difficult to find a distribution $\mathcal{D}$ in its 4th step.

There aren't numeric record in the original paper. Thus I will randomly generate parameters and then conduct numeric experiments to test the performance of these two algorithms

## File Strucutre

Experiment.ipynb: use a synthesis example to check the growing trend of Regret.

"Source":

+ agent.py: the source code of algorithm PrimalDualBwK
+ env.py
