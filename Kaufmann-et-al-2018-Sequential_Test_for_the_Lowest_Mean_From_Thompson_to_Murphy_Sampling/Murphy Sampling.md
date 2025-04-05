# Murphy Sampling

Denote $\psi(x; \eta,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\eta)^2}{2\sigma^2})$, $\Psi(x; \eta,\sigma^2)=\int_{-\infty}^x \psi(t; \eta,\sigma^2) dt$

## Single Random Variable

The main difficulties is to find a prior whose supper set is $(-\infty, \gamma)$ (For any low problem) or $(\gamma, +\infty)$​ (For any large problem.)

For Gaussian Conjugate, let's take prior $\nu$ as the following

+ Sample X from distribution with density $ \psi(x; \eta,\sigma^2)$
+ until $X\geq \gamma$

Then for any real value $x\leq \gamma$, we have $F_X(x)=0$.

For any real value $x>\gamma$, we have
$$
\begin{align*}
\Pr(X\leq x)=& \Pr(\exists n, X_i<\gamma,i=1,2,\cdots, n-1,\gamma < X_n\leq x)\\
= & \sum_{n=1}^{+\infty}\Pr( X_i<\gamma,i=1,2,\cdots, n-1,\gamma < X_n\leq x)\\
= & \sum_{n=1}^{+\infty}  \Psi(\gamma; \eta,\sigma^2)^{n-1} \Pr(\gamma < X_1\leq x)\\
= & \frac{\Pr(\gamma < X_1\leq x)}{1-\Psi(\gamma; \eta,\sigma^2)}\\
= & \frac{\Pr(X_1\leq x)-(\Pr(X_1\leq \gamma)}{1-\Psi(\gamma; \eta,\sigma^2)}\\
= & \frac{\Psi(x; \eta,\sigma^2)-\Psi(\gamma; \eta,\sigma^2)}{1-\Psi(\gamma; \eta,\sigma^2)}
\end{align*}
$$
where $X_i$ has density function $\psi(x; \eta,\sigma^2)$

Thus we can conclude $f_X(x)=\begin{cases} \frac{\psi(x; \eta,\sigma^2)}{1-\Psi(\gamma; \eta,\sigma^2)} & x>\gamma\\ 0 & x\leq \gamma\end{cases}$

Taking this as prior of $\mu$​, for $x>\gamma$, we have
$$
\begin{align*}
f_{\mu|\{X_i\}_{i=1}^{n} }(x)= & \frac{1}{1-\Psi(\gamma; \eta,\sigma^2)} \frac{\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\eta)^2}{2\sigma^2}) \left(\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi} \sigma}\exp(-\frac{(X_i-x)^2}{2\sigma^2})\right)}{\int_{\gamma}^{+\infty} \left(\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi} \sigma}\exp(-\frac{(X_i-t)^2}{2\sigma^2})\right)\frac{\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(t-\eta)^2}{2\sigma^2}}{1-\Psi(\gamma; \eta,\sigma^2)} dt}\\
= & \frac{\exp(-\frac{(x-\eta)^2}{2\sigma^2}) \left(\prod_{i=1}^{n}\exp(-\frac{(X_i-x)^2}{2\sigma^2})\right)}{\int_{\gamma}^{+\infty} \left(\prod_{i=1}^{n}\exp(-\frac{(X_i-t)^2}{2\sigma^2})\right)\exp(-\frac{(t-\eta)^2}{2\sigma^2}) dt}\\
= & \frac{\exp\left(-\frac{(n+1)x^2-2(\eta+\sum_{i=1}^n X_i)x}{2\sigma^2}\right)}{\int_{\gamma}^{+\infty} \exp\left(-\frac{(n+1)t^2-2(\eta+\sum_{i=1}^n X_i)t}{2\sigma^2}\right)  dt}\\
= & \frac{\psi(x; \frac{\eta+\sum_{i=1}^n X_i}{n+1},\frac{\sigma^2}{n+1})}{1-\Psi(\gamma; \frac{\eta+\sum_{i=1}^n X_i}{n+1},\frac{\sigma^2}{n+1})}
\end{align*}
$$
Or we can do it simpler
$$
f_{\mu|\{X_i\}_{i=1}^{n} }(x)\propto \exp(-\frac{(\mu-\eta)^2}{2\sigma^2})\prod_{i=1}^n \exp(-\frac{(X_i-\mu)^2}{2\sigma^2})\propto\exp\left(-\frac{(n+1)\mu^2-2(\eta+\sum_{i=1}^n X_i)\mu}{2\sigma^2}\right)
$$
Then we can conclude $f_{\mu | \{X_i\}_{i=1}^{n}}(x)=\begin{cases}\frac{\psi(x; \frac{\eta+\sum_{i=1}^n X_i}{n+1}, \frac{\sigma^2}{n+1})}{1-\Psi(x; \frac{\eta+\sum_{i=1}^n X_i}{n+1}, \frac{\sigma^2}{n+1})} & x > \gamma\\ 0 & x\leq \gamma\end{cases}$

We can sample the $\mu | \{X_i\}_{i=1}^{n}$ by 

+ Sample X from normal distribution $N(\frac{\eta+\sum_{i=1}^n X_i}{n+1}, \frac{\sigma^2}{n+1})$
+ If $X< \gamma$​, sample again.

Here we can take $\eta=\gamma$.

## Random Vector

Let's take prior $\mu\in \mathbb{R}^K$​ as the following

+ Sample $\mu$ from distribution with density $ \otimes_{a=1}^K\frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(x_a-\eta_a)^2}{2\sigma^2})$
+ until $\max_{a\in [K]}\mu_a\geq \mu_0$​

Then for any real value $x_a\leq \mu_0, a\in[K]$, we have $F_\mu(\mu_a\leq x_a)=0$​.

For any real vector such that $\max_{a\in[K]}x_a>\gamma$​, we have
$$
\begin{align*}
\Pr(\mu_a\leq x_a, \forall a)
= & \sum_{n=1}^{+\infty}\Pr( X_a^i<\mu_0,i\in [n-1], a\in [K]; X^n_a\leq x_a, \max_{a\in [K]}X_a^n > \mu_0)\\
= & \sum_{n=1}^{+\infty}\frac{\Pr(X^1_a\leq x_a, \max_{a\in [K]}X_a^1 > \mu_0)}{\prod_{a=1}^K\Psi(\mu_0; \eta_a,\sigma^2)^{n-1}} \\
= & \frac{\Pr(X^1_a\leq x_a, \forall a\in[K]; \max_{a\in [K]}X_a^1 > \mu_0)}{1-\prod_{a=1}^K\Psi(\mu_0; \eta_a,\sigma^2)}\\
= & \frac{\Pr(X^1_a\leq x_a, \forall a\in[K]) - \Pr(X^1_a\leq x_a, \forall a\in[K]; \max_{a\in [K]}X_a^1 \leq \mu_0) }{1-\prod_{a=1}^K\Psi(\mu_0; \eta_a,\sigma^2)}\\
= & \frac{\prod_{a=1}^K \Psi(x_a;\eta_a,\sigma^2) - \prod_{a=1}^K \Psi(\min\{x_a, \mu_0\};\eta_a,\sigma^2) }{1-\prod_{a=1}^K\Psi(\mu_0; \eta_a,\sigma^2)}
\end{align*}
$$
Thus, we can conclude the density is
$$
f(\{x_a\}_{a=1}^K) =\begin{cases}
\frac{\prod_{a=1}^K \psi(x_a;\eta_a,\sigma^2) - \prod_{a=1}^K \psi(x_a;\eta_a,\sigma^2)\mathbb{1}(x_a\leq \mu_0) }{1-\prod_{a=1}^K\Psi(\mu_0; \eta_a,\sigma^2)} & \max_{a\in [K]}\mu_a\geq \mu_0\\
0 & \max_{a\in [K]}\mu_a< \mu_0\\
\end{cases}
$$
Now, given $\{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K$ for each $a$, the conditional density of $\mu| \{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K $ is
$$
\begin{align*}
& f_{\mu| \{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K }(\{x_a\}_{a=1}^K)\\
\propto & \left(\prod_{a=1}^K \phi(x_a;\eta_a,\sigma^2) - \prod_{a=1}^K \phi(x_a;\eta_a,\sigma^2)\mathbb{1}(x_a\leq \mu_0)\right)\cdot \prod_{a=1}^K\exp\left(-\frac{\sum_{t=1}^{N_a} (X_{a,t}-x_a)^2}{2\sigma^2}\right)\\
\propto & \prod_{a=1}^K \exp\left(-\frac{(N_a+1)x_a^2 -(\eta_a+\sum_{t=1}^{n_a}X_{a,t})x_a}{2\sigma^2}\right)-\prod_{a=1}^K \exp\left(-\frac{(N_a+1)x_a^2 -(\eta_a+\sum_{t=1}^{n_a}X_{a,t})x_a}{2\sigma^2}\right)\mathbb{1}(x_a\leq \mu_0)\\
\end{align*}
$$
which is
$$
f_{r| \{\{X_{a,t}\}_{t=1}^{N_a}\}_{a=1}^K }(\{x_a\}_{a=1}^K)=\begin{cases}
\frac{\prod_{a=1}^K \psi(x_a;\frac{\eta_a+\sum_{t=1}^{N_a} X_{a, t}}{N_a+1},\frac{\sigma^2}{N_a+1}) - \prod_{a=1}^K \psi(x_a;\frac{\eta_a+\sum_{t=1}^{N_a} X_{a, t}}{N_a+1},\frac{\sigma^2}{N_a+1})\mathbb{1}(x_a\leq \mu_0) }{1-\prod_{a=1}^K\Psi(\mu_0; \frac{\eta_a+\sum_{t=1}^{N_a} X_{a, t}}{N_a+1},\frac{\sigma^2}{N_a+1})} & \max_{a\in [K]}x_a\geq \mu_0\\
0 & \max_{a\in [K]}x_a< \mu_0\\
\end{cases}
$$


We can sample the result by 

+ Sample $r_a$ from normal distribution $N(\frac{\eta_a+\sum_{t=1}^{N_a} X_{a, t}}{N_a+1}, \frac{\sigma^2}{N_a+1})$
+ until $\max_{a\in [K]}\mu_a\geq \mu_0$

Then given the vector $\{r_a\}_{a=1}^K$ such that $\max_{a\in [K]}r_a\geq \mu_0$, we pull arm $\arg\max_{a\in [K]}r_a$

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
