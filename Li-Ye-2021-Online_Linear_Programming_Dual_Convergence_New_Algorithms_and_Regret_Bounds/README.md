# README

This folder aims to reproduce part of the numeric experiments in Li & Ye 2021 *Online Linear Programming: Dual Convergence, New Algorithms, and Regret Bounds*

## File Structure

## Notes

In the first algorithm "No-Need-to-Learn", we should solve following stochastic programming problem with known distribution of $(r,\bold{a})$
$$
p^*=\arg\min_{p\ge 0} \ d^Tp +\mathbb{E}_{(r,\bold{a})\sim \mathcal{P}}\left[(r-\bold{a}^Tp)^+\right]
$$
Following the section 5.1 in the paper, we use SAA scheme with $10^6$ samples.
