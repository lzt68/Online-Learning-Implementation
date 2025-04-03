# Murphy Sampling

## Single Random Variable

The main difficulties is to find a prior whose supper set is $(-\infty, \gamma)$ (For any low problem) or $(\gamma, +\infty)$​ (For any large problem.)

For Gaussian Conjugate, let's take prior $\nu$ as the following

+ Sample X from distribution with density $ \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(x-\gamma)^2}{2\sigma^2})$
+ until $X\geq \gamma$

Then for any real value $x\leq \gamma$, we have $F_X(x)=0$.

For any real value $x>\gamma$, we have
$$
\begin{align*}
\Pr(X\leq x)=& \Pr(\exists n, X_i<\gamma,i=1,2,\cdots, n-1,\gamma < X_n\leq x)\\
= & \sum_{n=1}^{+\infty}\Pr( X_i<\gamma,i=1,2,\cdots, n-1,\gamma < X_n\leq x)\\
= & \sum_{n=1}^{+\infty}\frac{\Pr(\gamma < X_1\leq x)}{2^{n-1}} \\
= & 2\Pr(\gamma < X_1\leq x)\\
= & 2(\Pr(X_1\leq x)-(\Pr(X_1\leq \gamma))\\
= & 2\Pr(X_1\leq x)-1
\end{align*}
$$
where $X_i\sim N(\gamma, \sigma^2)$.

Thus we can conclude $f_X(x)=\begin{cases}\frac{2}{\sqrt{2\pi}\sigma} \exp(-\frac{(x-\gamma)^2}{2\sigma^2}) & x>\gamma\\ 0 & x\leq \gamma\end{cases}$

Taking this as prior of $\mu$, we have
$$
p(\mu|\{X_i\}_{i=1}^{n} )\propto \exp(-\frac{(\mu-\gamma)^2}{2\sigma^2})\prod_{i=1}^n \exp(-\frac{(X_i-\mu)^2}{2\sigma^2})\propto\exp\left(-\frac{(n+1)\mu^2-2(\gamma+\sum_{i=1}^n X_i)\mu}{2\sigma^2}\right)
$$
Then we can conclude $f_{\mu | \{X_i\}_{i=1}^{n}}(x)=\begin{cases}\frac{2}{\sqrt{2\pi}\frac{\sigma}{\sqrt{n+1}}} \exp(-\frac{(x-\frac{\gamma+\sum_{i=1}^n X_i}{n+1})^2}{2\cdot\frac{\sigma^2}{n+1}}) & x>\gamma\\ 0 & x\leq \gamma\end{cases}$

We can sample the result by 

+ Sample X from normal distribution $N(\frac{\gamma+\sum_{i=1}^n X_i}{n+1}, \frac{\sigma^2}{n+1})$
+ If $X< \gamma$, sample again.

## Random Vector

Let's take prior $\mu\in \mathbb{R}^K$​ as the following

+ Sample X from distribution with density $ \otimes_{a=1}^K\frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(x_a-\gamma)^2}{2\sigma^2})$
+ until $\max_{a\in [K]}\mu_a\geq \gamma$​

Then for any real value $x_a\leq \gamma, a\in[K]$, we have $F_\mu(\mu_a\leq x_a)=0$​.

For any real vector such that $\max_{a\in[K]}x_a>\gamma$​, we have
$$
\begin{align*}
\Pr(\mu_a\leq x_a, \forall a)
= & \sum_{n=1}^{+\infty}\Pr( X_a^i<\gamma,i\in [n-1], a\in [K]; X^n_a\leq x_a, \max_{a\in [K]}X_a^n > \gamma)\\
= & \sum_{n=1}^{+\infty}\frac{\Pr(X^1_a\leq x_a, \max_{a\in [K]}X_a^1 > \gamma)}{2^{(n-1)K}} \\
= & \frac{2^K}{2^K-1} \Pr(X^1_a\leq x_a, \forall a\in[K]; \max_{a\in [K]}X_a^1 > \gamma)
\end{align*}
$$
Then the derivative depends on the value of vector $\{x_a\}_{a\in [K]}$.

+ In the case that $K=2$, we have
  $$
  \begin{align*}
  & \frac{2^K}{2^K-1} \Pr(X^1_a\leq x_a, \forall a\in[K]; \max_{a\in [K]}X_a^1 > \gamma)\\
  = & \frac{4}{3} \Pr(X_1\leq x_1, X_1\leq x_2; \max\{X_1,X_2\} > \gamma)\\
  = & \frac{4}{3}\left(\Pr(X_1\leq x_1, X_1\leq x_2) - \Pr(X_1\leq x_1, X_1\leq x_2, \max\{X_1,X_2\} \leq  \gamma)\right)\\
  = & \frac{4}{3}\left(\Pr(X_1\leq x_1, X_1\leq x_2) - \Pr(X_1\leq \min\{x_1, \gamma\}, X_2\leq \min\{x_2, \gamma\}\leq  \gamma)\right)\\
  = & \frac{4}{3}\left(\Pr(X_1\leq x_1, X_1\leq x_2) - \int_{-\infty}^{\min\{x_1, \gamma\}}\phi(t_1) dt_1 \int_{-\infty}^{\min\{x_2, \gamma\}}\phi(t_2) dt_2 \right)\\
  = & \frac{4}{3}\left(\int_{-\infty}^{x_1}\phi(t_1) dt_1 \int_{-\infty}^{x_2}\phi(t_2) dt_2 - \int_{-\infty}^{\min\{x_1, \gamma\}}\phi(t_1) dt_1 \int_{-\infty}^{\min\{x_2, \gamma\}}\phi(t_2) dt_2 \right)\\
  \end{align*}
  $$
  where $\phi(t)=\frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(t-\gamma)^2}{2\sigma^2})$​.

+ In the case that $K$​ takes general value, we have
  $$
  \frac{2^K}{2^K-1} \Pr(X^1_a\leq x_a, \forall a\in[K]; \max_{a\in [K]}X_a^1 > \gamma) =\frac{2^K}{2^K-1}\left(\prod_{a=1}^K \int_{-\infty}^{x_a}\phi(t_a) dt_a - \prod_{a=1}^K \int_{-\infty}^{\min\{x_a,\gamma\} }\phi(t_a) dt_a\right)
  $$

Then, we can conclude the density function of $\mu$ is
$$
\frac{2^K}{2^K-1}\left(\prod_{a=1}^K \phi(x_a) - \prod_{a=1}^K \phi(x_a)\mathbb{1}(x_a>\gamma)\right)
$$


> $$
> \sum_{n=1}^{+\infty}\frac{1}{(2^K)^{n-1}} = \frac{1}{1-\frac{1}{2^K}} = \frac{2^K}{2^K-1}
> $$
>

Now, given $\{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K$ for each $a$, the conditional density of $\mu| \{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K $ is
$$
\begin{align*}
& f_{\mu| \{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K }(\{x_a\}_{a=1}^K)\\
\propto & \frac{2^K}{2^K-1}\left(\prod_{a=1}^K \phi(x_a) - \prod_{a=1}^K \phi(x_a)\mathbb{1}(x_a>\gamma)\right)\cdot \prod_{a=1}^K\exp\left(-\frac{\sum_{t=1}^{n_a} (X_{a,t}-x_a)^2}{2\sigma^2}\right)\\
\propto & \prod_{a=1}^K \exp\left(-\frac{(n_a+1)x_a^2 -(\gamma+\sum_{t=1}^{n_a}X_{a,t})x_a}{2\sigma^2}\right)-\prod_{a=1}^K \exp\left(-\frac{(n_a+1)x_a^2 -(\gamma+\sum_{t=1}^{n_a}X_{a,t})x_a}{2\sigma^2}\right)\mathbb{1}(x_a>\gamma)\\
\end{align*}
$$
We can sample the result by 

+ Sample $\mu_a$ from normal distribution $N(\frac{\gamma+\sum_{t=1}^{n_a} X_{a, t}}{n_a+1}, \frac{\sigma^2}{n+1})$
+ until $\max_{a\in [K]}\mu_a\geq \gamma$

## Bayesian Rule

If $\mu\sim N(0, 1)$, and we have collected samples $\{X_i\}_{i=1}^{n}$

we have
$$
p(\mu|\{X_i\}_{i=1}^{n} )\propto \exp(-\frac{\mu^2}{2})\prod_{i=1}^n \exp(-\frac{(X_i-\mu)^2}{2})\propto\exp\left(-\frac{(n+1)\mu^2-2(\sum_{i=1}^n X_i)\mu}{2}\right)
$$
We can assert $\mu|\{X_i\}_{i=1}^{n} \sim N\left(\frac{\sum_{i=1}^n X_i}{n+1},\frac{1}{n+1}\right)$



If $\mu\sim N(1, \sigma^2)$, and we have collected samples $\{X_i\}_{i=1}^{n}$

we have
$$
p(\mu|\{X_i\}_{i=1}^{n} )\propto \exp(-\frac{(\mu-1)^2}{2\sigma^2})\prod_{i=1}^n \exp(-\frac{(X_i-\mu)^2}{2\sigma^2})\propto\exp\left(-\frac{(n+1)\mu^2-2(\sum_{i=1}^n X_i+1)\mu}{2\sigma^2}\right)
$$
We can assert $\mu|\{X_i\}_{i=1}^{n} \sim N\left(\frac{1+\sum_{i=1}^n X_i}{n+1},\frac{\sigma^2}{n+1}\right)$
