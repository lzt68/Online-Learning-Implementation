# README

This folder tried to reproduce the figure 2 in the Abbasi-yadkori et al.2011, which involves

+ OFUL algorithm, (O: Optimism, U: Uncertainty, F: Face, L: Linear)
  https://sites.ualberta.ca/~szepesva/papers/linear-bandits-NeurIPS2011.pdf

+ Confidence Ball algorithm, from Dani et al. 2008

  https://repository.upenn.edu/statistics_papers/101/

## File Structure

"Source": Include the source file of OFUL algorithm, Confidence Ball algorithm and the implementation of environment

+ agent.py: Implement the OFUL algorithm and the Confidence Ball 2 algorithm
+ env.py: Implement the environment

Demo.ipynb: Demo for how to use the source file to conduct numeric experiments

Proof Structure.pptx: Proof structure of paper Abbasi-yadkori et al.2011, showing up as a figure, which might help you understand their idea.

## Find the point with maximal $l_2$ norm on the ellipsoid

The major challenge of implementing this topic is the following step
$$
(X_t, \tilde{\theta}_t)=\arg\max_{(x,\theta)\in D_t\times C_{t-1}} <x,\theta>
$$
Or we can replace $\max$ with $\min$, there isn't significant difference.

In Abbasi-yadkori et al.2011, $D_t$ is exogenous and can be arbitrary, and $C_t$ is the confidence space, depending on our observations. Even when both of them are convex set, we might still be unable to derive the exact optimal point of this optimization problems. 

**I doubt there isn't a general solution of solving this problem**

In the figure 2 of Abbasi-yadkori et al.2011, we consider $2$-dimension linear bandits, with assuming **the parameters vector and actions are from the unit ball**. Then we can conclude
$$
\begin{align*}
&(X_t, \tilde{\theta}_t)=\arg\max_{(x,\theta)\in D_t\times C_{t-1}} <x,\theta>\\
\Rightarrow &\tilde{\theta}_t = \arg\max_{\theta\in C_{t-1}} <\frac{\theta}{\|\theta\|_2},\theta>=\arg\max_{\theta\in C_{t-1}}\|\theta\|_2, X_t=\frac{\tilde{\theta}_t}{\|\tilde{\theta}_t\|_2}\\
\end{align*}
$$
In the case of 2-d, $C_{t}=\{x:(x-x_0)^TA_t(x-x_0)=\beta_t\}$, Denote $A_t=U \Lambda U^T$, where $\Lambda=\left[\begin{array}{l}\lambda_1&0\\ 0&\lambda_2 \end{array}\right]$, then 
$$
\begin{align*}
&(x-x_0)^TA_t(x-x_0)=\beta_t\\
\Leftrightarrow& (x-x_0)^TU \left[\begin{array}{l}\lambda_1&0\\ 0&\lambda_2 \end{array}\right] U^T(x-x_0)=\beta_t
\end{align*}
$$
Let $U^T(x-x_0)=\left[\begin{array}{c}\frac{\sqrt{\beta_t}\sin t}{\sqrt{\lambda_1}}\\ \frac{\sqrt{\beta_t}\cos t}{\sqrt{\lambda_2}}\end{array}\right]$, which is $x=U\left[\begin{array}{c}\frac{\sqrt{\beta_t}\sin t}{\sqrt{\lambda_1}}\\ \frac{\sqrt{\beta_t}\cos t}{\sqrt{\lambda_2}}\end{array}\right]+x_0$, then we have
$$
\begin{align*}
\|x\|_2^2&=\|x_0\|_2^2 + 2\left[\begin{array}{c}\frac{\sqrt{\beta_t}\sin t}{\sqrt{\lambda_1}}& \frac{\sqrt{\beta_t}\cos t}{\sqrt{\lambda_2}}\end{array}\right]U^Tx_0+\frac{\beta_t \sin^2t}{\lambda_1}+\frac{\beta_t \cos^2t}{\lambda_2}
\end{align*}
$$
In my implementation I first choose 10 points on the ellipsoid, and then use the point with maximal l2-norm as the initial point of the scipy.optimize pacakge.
