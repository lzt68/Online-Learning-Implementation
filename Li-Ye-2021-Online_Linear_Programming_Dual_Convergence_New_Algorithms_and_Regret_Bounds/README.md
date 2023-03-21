# README

This folder aims to reproduce part of the numeric experiments in Li & Ye 2021 *Online Linear Programming: Dual Convergence, New Algorithms, and Regret Bounds*

## File Structure

## Notes of No-Need-to-Learn

In the first algorithm "No-Need-to-Learn", we should solve following stochastic programming problem with known distribution of $(r,\bold{a})$
$$
p^*=\arg\min_{p\ge 0} \ d^Tp +\mathbb{E}_{(r,\bold{a})\sim \mathcal{P}}\left[(r-\bold{a}^Tp)^+\right]
$$
where $p^*, d, a\in \mathbb{R}^m$.

In the numeric experiments, all the entries of $d$ are the same. And further they always assume the entries of $a$ are i.i.d distribution. **That means the object function are symmetric regarding p. We can prove $p^*_1=p^*_2=\cdots=p^*_m$**.

Denote $f(p_1, p_2,\cdots,p_m) = d^T\left[\begin{array}{c}p_1\\p_2\\ \vdots\\ p_m\end{array}\right]+\mathbb{E}_{(r,\bold{a})\sim \mathcal{P}}\left[(r-\bold{a}^T\left[\begin{array}{c}p_1\\p_2\\ \vdots\\ p_m\end{array}\right])^+\right]$, it is easy to see for any permutation $\sigma$ of $[m]$, we have
$$
f(p_1, p_2,\cdots,p_m) = f(p_{\sigma(1)}, p_{\sigma(2)},\cdots,p_{\sigma(m)})
$$
And it is easy to check $f$ is a convex function. Thus for any $p_1, p_2,\cdots, p_m \ge 0$, we have
$$
\begin{align*}
&f(\frac{\sum_{i=1}^m p_i}{m}, \cdots, \frac{\sum_{i=1}^m p_i}{m})\\
\le & \frac{f(p_1, p_2, \cdots,m)+f(p_m, p_1, p_2, \cdots, p_{m-1})+\cdots+f(p_2, p_3,\cdots, p_{m}, p_1)}{m}\\
= & f(p_1, p_2,\cdots,p_m)
\end{align*}
$$
We completed the proof.

So it suffices to minimize following function
$$
p^*=\arg\min_{p\ge 0} \ pmd +\mathbb{E}_{(r,\bold{a})\sim \mathcal{P}}\left[(r-p\sum_{i=1}^m a_i)^+\right], \ s.t. \ p\ge0
$$
where $p\in \mathbb{R}$

In general, consider function $dp  +\mathbb{E}\max \{r-ap, 0\}$, and denote $f$ as the pdf of $a$, $g$ as the pdf of $r$  
Define $A=\int_{-\infty}^{\frac{r}{p}} f(a) da, B=\int_{-\infty}^{\frac{r}{p}} af(a) da$
$$
\begin{align*}
&dp+\mathbb{E}\max \{r-ap, 0\}\\
=&dp+\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}  \max \{r-ap, 0\} f(a)g(r) da dr\\
=&dp+\int_{-\infty}^{+\infty}\int_{-\infty}^{\frac{r}{p}} (r-ap)f(a) g(r)da dr\\
=&dp+\int_{-\infty}^{+\infty}(rA - pB)g(r) dr\\
=&\int_{-\infty}^{+\infty}(rA +(d-B) p)g(r) dr
\end{align*}
$$
Thus
$$
\begin{align*}
&\frac{\partial \left(dp+\mathbb{E}\max \{r-ap, 0\}\right)}{\partial p }\\
=&\int_{-\infty}^{+\infty}\left(r \frac{\partial A}{\partial p} + \frac{\partial (d-B) p}{\partial p}\right) g(r) dr\\
=&\int_{-\infty}^{+\infty}\left(r f(\frac{r}{p})(-\frac{r}{p^2}) + \frac{\partial (d-B)}{\partial p}p+(d-B)\right) g(r) dr\\
=&\int_{-\infty}^{+\infty}\left(r f(\frac{r}{p})(-\frac{r}{p^2}) + \frac{\partial (-B)}{\partial p}p+(d-B)\right) g(r) dr\\
=&\int_{-\infty}^{+\infty}\left(r f(\frac{r}{p})(-\frac{r}{p^2}) - \frac{r}{p}f(\frac{r}{p})(-\frac{r}{p^2})p+(d-B)\right) g(r) dr\\
=&\int_{-\infty}^{+\infty} \left(d-B\right) g(r) dr
\end{align*}
$$
which implies if $dm \ge \mathbb{E}\sum_{i=1}^m a_i$, the optimal solution $p^*= 0$. You can check the Random Input I in the section 5.1 fits in this case.

In the section 5.1 in the paper, the author suggests using SAA scheme with $10^6$ samples. Due to the computation limit, I used SGD to solve the optimal $p^*$

## Notes on Simplified-Dynamic-Learning

In the second step of Simplified-Dynamic-Learning, we need to find $\delta\in (1, 2]$ and $L\in \mathbb{Z}$, s.t. $\lfloor \delta^L \rfloor = n$

Then we can take $L=\lceil\log_2 n\rceil$, $\delta=n^{\frac{1}{L}}$, since $n^{\frac{1}{L}}\le n^{\frac{1}{\log_2 n}}=2$


$$
p^*_k=\arg\min_p \sum_{i=1}^m d_ip_i+\frac{1}{t_k}\sum_{j=1}^{t_k}\left(r_j-\sum_{i=1}^m a_{ij} p_i\right)^+, s.t. p\ge0
$$
