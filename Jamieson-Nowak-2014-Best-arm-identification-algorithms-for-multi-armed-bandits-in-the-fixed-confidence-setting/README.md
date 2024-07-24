# README

We aim to reproduce the figure 1 in Jamieson & Nowak 2014 "Best-arm identification algorithms for multi-armed bandits in the fixed confidence setting"

## Experiment Setting

+ 6 arms, with mean reward $\{1, \frac{4}{5}, \frac{3}{5}, \frac{2}{5}, \frac{1}{5}, 0\}$
+ The reward follows Gaussian distribution with variance $\frac{1}{4}$
+ 5000 independent experiments for each algorithm

## File Structure

"Source": The source file of algorithm and environment

+ agent.py: the source file of Action Elimination, UCB, LUCB algorithm
+ env.py: the source file of environment that always generate Gaussian reward

Experiment.ipynb: Reproduce the 3 plots in figure 1

## Supplement Details

In the fixed confidence setting, the stopping time of a single experiment is uncertain. In  Experiment.ipynb, I set the length of each action history is at most 75\*int(H1), (H1 is roughly 36). If the agent stops before 75\*int(H1), I use the prediction of the agent as the following action until we achieve round 75\*int(H1).

I am not sure whether I correctly implement the idea of the paper, but at least we got similar figures in Experiment.ipynb.

