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
In the algorithm, if $\max_{1\leq a\leq K}\hat{\mu}_{a,t} > \mu_0$, we have
$$
\mathcal{C}_t:=\{\vec{\mu'}: \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\hat{\mu}_{a,t})^2}{2}\leq \log C +10\log(t-1)\}\\
I_t=\cup_{\mu'\in\mathcal{C}_t}\{\arg\max_i \mu'_i\}\\
w_t = e_{i_t}\\
\mathcal{D}_t:=\{\vec{\mu'}: \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\mu_a)^2}{2}\leq \log(\frac{Ct^2}{\delta})\}\\
$$
where $C\geq e\sum_{t=1}^{+\infty}(\frac{e}{K})\frac{(\log^2(Ct^2)\log t)^K}{t^2}$. 

### Determine the elements in $I_t$

As we use Gaussian Distribution, here we don't consider the case $\hat{\mu}_{a,t}=\mu_0$ for some $a\in[K]$ and $t\in \mathbb{N}$, also we assume all the inequalities strictly hold

+ If $\hat{\mu}_{a,t} < \mu_0$ holds for all $a$, then $I_t=[K]$, 
  and the $w^*(\hat{\mu}_{t-1}, \neg \text{none})$ is determined by $w_a^*=\frac{\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}{\sum_{i=1}^K\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}=\frac{\frac{1}{(\hat{\mu}_{a,t}-\mu_0)^2}}{\sum_{i=1}^K\frac{1}{(\hat{\mu}_{a,t}, \mu_0)^2}}$, and we can see $D(\vec{w},\vec{\mu},\neg \text{none})=\min w_a\frac{(\mu_a-\mu_0)^2}{2}$,  $D(\vec{\mu},\neg \text{none})=\frac{1}{\sum_{a=1}^K\frac{2}{(\mu_a-\mu_0)^2}}$.

+ If $\max_{1\leq a\leq K}\hat{\mu}_{a,t} > \mu_0$, without loss of generality, assume $\hat{\mu}_{1,t} \geq \hat{\mu}_{2,t}\geq\cdots \geq \hat{\mu}_{m,t} > \mu_0 > \hat{\mu}_{m+1, t}\geq \cdots\geq \hat{\mu}_{K, t}$, then

  + $1\in i_F(\hat{\mu}_t)\subset I_t$

  + For $2\leq a\leq m$, if $\sum_{i=1}^{a-1}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\hat{\mu}_{a,t})^2}{2}< \log C +10\log(t-1)$, then
    $a\in i_F(\underbrace{\hat{\mu}_{a,t},\cdots,\hat{\mu}_{a,t}}_{a-1},\hat{\mu}_{a,t}+\Delta,\hat{\mu}_{a+1,t},\cdots,\hat{\mu}_{K,t})\subset I_t$ for some positive $\Delta$,

  + For $m+1\leq a\leq K$, if $\left(\sum_{i=1}^{m}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_0)^2}{2}\right) + N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_0)^2}{2}< \log C +10\log(t-1)$, then

    $a\in i_F(\underbrace{\mu_0,\cdots,\mu_0}_{m},\hat{\mu}_{m+1,t},\cdots,\hat{\mu}_{a-1,t}, \underbrace{\mu_0+\Delta}_{a \text{ th entry} },\hat{\mu}_{a+1,t},\cdots,\hat{\mu}_{K,t})\subset I_t$ for some positive $\Delta$,

  The reason is, if for some $\vec{\mu}'$, $i_F(\vec{\mu'})=a\in \{2,\cdots, K\}$, the above constructed instance vector would be arbitrarily close to the the $\inf_{\vec{\mu'}: i_F(\vec{\mu'})=a} \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\hat{\mu}_{a,t})^2}{2}$, by setting small enough $\Delta$.
   <font color=red>This statement might be incorrect, we need to rigorously derive the explicit value of $\inf_{\vec{\mu'}: i_F(\vec{\mu'})=a} \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\hat{\mu}_{a,t})^2}{2}$.</font>

By the sticky pulling rule, denote $I_t=\{a_1, a_2,\cdots, a_{k_t}\}$ with $a_1 < a_2 <\cdots<a_{k_t}$, we always take $i_t=a_1$.

## Determine $i_t$

By the sticky pulling rule, we would select $i_t$ as the smallest arm index in the $I_t$. As the only usage of $I_t$ is to find out $i_t$ for the next step, there is no need to figure out all the elements in $I_t$. We should iterate arm 1 to $K$ to find out the arm with smallest index that is in $I_t$.

## Determine $\vec{w}_t$

+ If $\hat{\mu}_{a,t} < \mu_0$ holds for all $a$, we take $w_a^*=\frac{\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}{\sum_{i=1}^K\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}=\frac{\frac{1}{(\hat{\mu}_{a,t}-\mu_0)^2}}{\sum_{i=1}^K\frac{1}{(\hat{\mu}_{a,t}, \mu_0)^2}}$
+ If $\max_{1\leq a\leq K}\hat{\mu}_{a,t} > \mu_0$
  + If $i_t\in i^*(\hat{\mu_t})$, we take $\vec{w}_t=e_{i_t}$
  + If $i_t\notin i^*(\hat{\mu_t})$, $\vec{w}_t$ can be any vector in $\Delta_K$, as $\arg\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \neg i_t}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}=\arg\sup_{\vec{w}\in \Delta_K}0$. Here we take $\vec{w}_t=(\frac{1}{K},\frac{1}{K},\cdots,\frac{1}{K})$.

## Determine whether to stop

The stopping rule in the algorithm is $\exist i\in [K]\cup\{\text{none}\}$, such that $\{\vec{\mu'}: D(N_t,\hat{\mu}_t,\vec{\mu'})\leq \log\frac{Ct^2}{\delta}\} \cap \neg i=\emptyset$.

+ For the case $\max_{1\leq a\leq K}\hat{\mu}_{a,t} > \mu_0$, denote $a_0$ as the arm with largest empirical mean reward we stop if
  $$
  N_{a_0}(t-1)\frac{(\hat{\mu}_{a_0, t}-\mu_0)^2}{2} > \log\frac{Ct^2}{\delta}
  $$

+ For the case $\max_{1\leq a\leq K}\hat{\mu}_{a,t} \leq \mu_0$, we stop if
  $$
  N_{a}(t-1)\frac{(\hat{\mu}_{a, t}-\mu_0)^2}{2}>\log\frac{Ct^2}{\delta}
  $$
   holds for all $a\in[K]$.

# File Structure

