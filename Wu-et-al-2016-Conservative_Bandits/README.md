# README

This folder aims to implement the **Conservative UCB** algorithm in *Wu et al. 2013 Conservative_Bandits*. Link of the paper: http://arxiv.org/abs/1602.04282. 

In this work, the authors use the the confidence interval of cumulative rewards to avoid violation of safety constraint. Regarding the choice of the length of lower bound, the author gave two possible choices.
$$
\begin{align*}
\Delta_i(t)=&\sqrt{\frac{2\log\frac{KN_i(t-1)^3}{\delta}}{N_i(t-1)}},\\
\Delta_i(t)=&\sqrt{\frac{\log \max\{3, \log\frac{K}{\delta}\} + \log (\frac{2e^2K}{\delta})+ \frac{\frac{K}{\delta}(1+\log\frac{K}{\delta})}{(\frac{K}{\delta}-1)\log\frac{K}{\delta}}\log\log (1+N_i(t-1))}{N_i(t-1)}}
\end{align*}
$$
In fact, there are still other choices. Instead of approximating the lower bound of each arm and then sum them up, we can turn to approximate the lower bound of overall reward we observed so far.

In the implementation, I **didn't** reproduce the experiments mentioned in the paper. But I implemented three different algorithms with different choices of length of confidence interval.

1. Conservative UCB (proposed by the paper)
2. Conservative UCB with the above 2nd $\Delta_i(t)$ (proposed by the paper)
3. Modified Conservative UCB (proposed by me)

Then I conducted experiments to see the performance of these algorithms. More analysis on the volation prob of my proposed algorithm are still required.

## File Strucutre

Experiment.ipynb: use a synthesis example to check the growing trend of Regret.

"Source":

+ agent.py: the source code of algorithm Conservative_UCB, Conservative_UCB_rad2, Conservative_UCB_Overall_Lower_Bound
+ env.py: the source code of the environment
