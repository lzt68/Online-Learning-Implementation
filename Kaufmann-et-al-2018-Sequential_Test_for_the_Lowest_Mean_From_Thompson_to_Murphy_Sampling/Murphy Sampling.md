# Murphy Sampling

The main difficulties is to find a prior whose supper set is $(-\infty, \gamma)$ (For any low problem) or $(\gamma, +\infty)$ (For any large problem.)

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
