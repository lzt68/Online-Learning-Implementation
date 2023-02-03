# README

This folder tried to reproduce the figure 2 in the Abbasi-yadkori et al.2011, which involves

+ OFUL algorithm, (O: Optimism, U: Uncertainty, F: Face, L: Linear)
  https://sites.ualberta.ca/~szepesva/papers/linear-bandits-NeurIPS2011.pdf

+ Confidence Ball algorithm, from Dani et al. 2008

  https://repository.upenn.edu/statistics_papers/101/

The major challenge of implementing this topic is the following step
$$
(X_t, \tilde{\theta}_t)=\arg\max_{(x,\theta)\in D_t\times C_{t-1}} <x,\theta>
$$
Or we can replace $\max$ with $\min$, there isn't significant difference.

In Abbasi-yadkori et al.2011, $D_t$ is exogenous and can be arbitrary, and $C_t$ is the confidence space, depending on our observations. Even when both of them are convex set, we might still be unable to derive the exact optimal point of this optimization problems. 

**I doubt there isn't a general solution of solving this problem**

In the figure 2 of Abbasi-yadkori et al.2011, we consider $2-d$ linear bandits, with assuming **the parameters vector and actions are from the unit ball**. Then we can conclude
$$
\begin{align*}
&(X_t, \tilde{\theta}_t)=\arg\max_{(x,\theta)\in D_t\times C_{t-1}} <x,\theta>\\
\Rightarrow &\tilde{\theta}_t = \arg\max_{\theta\in C_{t-1}} <\frac{\theta}{\|\theta\|_2},\theta>=\arg\max_{\theta\in C_{t-1}}\|\theta\|_2, X_t=\frac{\tilde{\theta}_t}{\|\tilde{\theta}_t\|_2}\\
\end{align*}
$$
Then this is a second-order-programming problem.
