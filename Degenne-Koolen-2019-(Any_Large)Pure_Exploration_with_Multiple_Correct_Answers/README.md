# README

This folder aims to apply the algorithm, in Degenne \& Koolen 2019 Pure Exploration with Multiple Correct Answers, to the *Any Low Problem*, stated as Example 1 in the paper. 

Paper Link: https://proceedings.neurips.cc/paper_files/paper/2019/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html

For convenience, here we consider **Any Large Problem**, which is essentially the same as the Example 1 in the paper. The problem formulation is as follows. For an instance $\nu$ with K arms, threshold $\mu_0$ (sometimes we use notation $\xi$ instead), and corresponding mean reward vector $\{r^{\nu}_{a}\}_{a=1}^K$​, we define "success" as

+ output an arm $a$ whose $r_a^{\nu}>\mu_0$, if $\max_{1\leq a\leq K}r_a^{\nu} > \mu_0$
+ output "none", if $\max_{1\leq a\leq K}r_a^{\nu} < \mu_0$

We want to design an algorithm which take confidence level $\delta$ as input, such that for any instance $\nu$, the algorithm can achieve success with probability $1-\delta$, while consuming as small pulling times as possible.

We only work on instances whose maximum mean reward is not $\mu_0$​, and unit variance Gaussian Distribution

## Calculation - Simplification

As we only consider unit variance Gaussian Distribution, for any mean reward vectors $\vec{\mu},\vec{\lambda}$ and pulling times vector $\vec{w}$, we have
$$
\neg i:=\{\vec{\mu}\in\mathcal{M}: i\notin i^*(\vec{\mu})\}, \text{all the instances whose answer set not include arm } i\\
D(\vec{w}, \vec{\mu}, \vec{\lambda}):=\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}\\
D(\vec{w}, \vec{\mu}, \Lambda):=\inf_{\vec{\lambda}\in \Lambda}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}\\
D(\vec{\mu}, \Lambda):=\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \Lambda}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}\\
D(\vec{\mu}, \Lambda):=\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \Lambda}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}\\
D(\vec{\mu}):=\max_{1\leq i\leq K}\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \neg i}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}\\
i_F(\vec{\mu}):=\arg\max_{i}\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \neg i}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}
$$
Consider vector $\vec{\mu}$ whose $\mu_1\geq \cdots\geq \mu_m > \mu_0\geq \mu_{m+1}\geq \cdots\geq \mu_K$​, easy to see $i^*(\vec{\mu})=[m]$. For $i\in [m]$,
$$
D(\vec{w},\vec{\mu},\neg i)=w_i \frac{(\mu_i-\mu_0)^2}{2}, D(\mu, \neg i)=\frac{(\mu_i-\mu_0)^2}{2}, \\D(\vec{\mu})=\frac{(\mu_1-\mu_0)^2}{2},i_F(\vec{\mu})=1=\arg\max_{1\leq a\leq K}\mu_a\\
w^*(\mu,\neg i )=\arg\max_{w} D(\vec{w},\vec{\mu}, \neg i)=e_i
$$
In the algorithm, we have
$$
\mathcal{C}_t:=\{\vec{\mu'}: \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\mu_a)^2}{2}\leq \log C +10\log(t-1)\}\\
I_t=\cup_{\mu'\in\mathcal{C}_t}\{\arg\max_i \mu'_i\}\\
w_t = e_{i_t}\\
\mathcal{D}_t:=\{\vec{\mu'}: \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\mu_a)^2}{2}\leq \log(\frac{Ct^2}{\delta})\}\\
$$
where $C\geq e\sum_{t=1}^{+\infty}(\frac{e}{K})\frac{(\log^2(Ct^2)\log t)^K}{t^2}$.

## File Structure

