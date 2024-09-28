# README

This folder aims to apply the algorithm, in Degenne \& Koolen 2019 Pure Exploration with Multiple Correct Answers, to the *Any Low Problem*, stated as Example 1 in the paper. 

Paper Link: https://proceedings.neurips.cc/paper_files/paper/2019/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html

For convenience, here we consider **Any Large Problem**, which is essentially the same as the Example 1 in the paper. The problem formulation is as follows. For an instance $\nu$ with K arms, threshold $\mu_0$ (sometimes we use notation $\xi$ instead), and corresponding mean reward vector $\{r^{\nu}_{a}\}_{a=1}^K$​, we define "success" as

+ output an arm $a$ whose $r_a^{\nu}>\mu_0$, if $\max_{1\leq a\leq K}r_a^{\nu} > \mu_0$
+ output "none", if $\max_{1\leq a\leq K}r_a^{\nu} < \mu_0$

We want to design an algorithm which take confidence level $\delta$ as input, such that for any instance $\nu$, the algorithm can achieve success with probability $1-\delta$, while consuming as small pulling times as possible.

We only work on instances whose maximum mean reward is not $\mu_0$​, and unit variance Gaussian Distribution.

# Supplement Details for the Algorithm

## Simplification of the notations

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

## Calculation of  $i_F(\vec{\mu})$

From the example 1 in the page 3, we know

+ If $\max_{1\leq a\leq K} \mu_a > \mu_0$, then $i_F(\vec{\mu})=\arg\max_a \mu_a$
+ If $\max_{1\leq a\leq K} \mu_a \leq \mu_0$, for any arm $i\in [K]$, $\vec{\mu}\in \neg i$, thus $\inf_{\vec{\lambda}\in \neg i}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}=0$ holds for any $i\in [K]$.
  And $i_F(\vec{\mu})=\arg\max_a \sup_{\vec{w}_k\in \Delta_K} 0 =[K]$.
  Thus, $i_F(\vec{\mu})=[K]$.

## Judge whether an arm $a$ is in $I_t$

+ If $\hat{\mu}_{i,t} < \mu_0$ holds for all $i$, then $I_t=[K]$, further $a\in I_t$.
+ If $\sum_{i=1}^KN_i(t-1)\frac{(\hat{\mu}_{i,t}-\min\{\mu_0, \hat{\mu}_{i,t}\})^2}{2} < \log f(t-1)$, then vector $\vec{\mu}'$ whose $a$-th entry is $\mu_a' = \min\{\hat{\mu}_{i,t},\mu_0\}$ is in $\mathcal{C}_t:=\left\{\vec{\mu'}: \sum_{a=1}^K N_a(t-1)\frac{(\mu'_a-\hat{\mu}_{a,t})^2}{2}\leq \log f(t-1)\right\}$. 
  Since $i_F(\vec{\mu}')=[K]$, we can conclude $I_t=[K]$​.
+ If $a=\arg\max_{1\leq i\leq K} \hat{\mu}_{i,t}$, then $a\in I_t$.

For an arm $a$ which is not the arm with maximum empirical mean reward, we can consider finding a vector $\vec{\mu'}\in S_a:=\left\{\vec{\mu}: a=\arg\max_{1\leq i\leq K}\mu_i, \mu_a > \mu_0\right\}$. If $\inf_{\vec{\mu'}\in S_a}\sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2} < \log f(t-1)$, we can conclude $a\in I_t$. Then the problem reduce to the calculation of $\inf_{\vec{\mu'}\in S_a}\sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}$. Notice that
$$
\begin{align*}
& \inf_{\vec{\mu'}\in S_a}\sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}\\
= & \inf_{\mu_a'\geq \mu_0}\inf_{\mu_a'\geq \mu_i', \forall i\neq a} \sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}
\end{align*}
$$
We can firstly solve the sub-problem $\inf_{\mu_a'\geq \mu_i', \forall i\neq a} \sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}=N_a(t-1)\frac{(\mu_a'-\hat{\mu}_{a,t})^2}{2} + \sum_{i=1, i\neq a}^K\inf_{\mu_i' \leq \mu_a'}N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}$, given the choice of $\mu_a'$. 

+ For arm index $j$ whose $\hat{\mu}_{j,t} < \mu_a'$, we know the optimal value must be $\mu_j'^*=\hat{\mu}_{j,t}$.

  > By the fact that $\inf_{\mu_j' \leq \mu_a'}N_j(t-1)\frac{(\mu'_j-\hat{\mu}_{j,t})^2}{2}\stackrel{\text{take }\mu_j'=\hat{\mu}_{j,t}}{\geq} 0$.

+ For arm index $j$ whose $\hat{\mu}_{j,t} \geq \mu_a'$, we know the optimal value must be $\mu_j'^*=\mu_a'$.

  > By the fact that $\inf_{\mu_j' \leq \mu_a'}N_j(t-1)\frac{(\mu'_j-\hat{\mu}_{j,t})^2}{2}\geq N_j(t-1)\frac{(\mu'_a-\hat{\mu}_{j,t})^2}{2}$.

Thus, we get $\inf_{\mu_a'\geq \mu_i', \forall i\neq a} \sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}=N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i\neq a}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$​. 

To solve $\min_{\mu_a' \geq \mu_0}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i\neq a}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$, we can see

+ If $\hat{\mu}_{a,t}\geq \mu_0$, we can conclude the optimal $\mu_a'$ must lie in the interval $(\hat{\mu}_{a,t}, \max_{1\leq i\leq K}\hat{\mu}_{i,t})$. Then we only need to solve $\min_{\mu_a' \geq \hat{\mu}_{a,t}}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$.
+ If $\hat{\mu}_{a,t} <  \mu_0$, we can conclude the optimal $\mu_a'$ must lie in the interval $(\mu_0, \max_{1\leq i\leq K}\hat{\mu}_{i,t})$. Then we only need to solve $\min_{\mu_a' \geq \mu_0}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \mu_0}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$

**It seems very hard to derive an explicit minimum point of the function.** But easy to see this is a convex function regarding $\mu_a'$, then we can use binary search to approximate the minimum point.

## Determine $i_t$

By the sticky pulling rule, we would select $i_t$ as the smallest arm index in the $I_t$. As the only usage of $I_t$ is to find out $i_t$ for the next step, there is no need to figure out all the elements in $I_t$. We should iterate arm 1 to $K$ to find out the arm with smallest index that is in $I_t$.

## Specify the value of Constant $C$
In the Theorem 10, the authors require to take $\beta(t,\delta)=\log\frac{Ct^2}{\delta}$ with $C\geq e\sum_{t=1}^{+\infty}(\frac{e}{K})^K\frac{\left(\log (Ct^2))^2\log t\right)^K}{t^2}$, without delivering an exact value of $C$. This constant also occurs at the Lemma 14 at page 7, which take $f(t)=Ct^{10}$.

If we take $C=1$, we have
$$
\begin{align*}
& e\sum_{t=1}^{+\infty}(\frac{e}{K})^K\frac{\left(\log (Ct^2))^2\log t\right)^K}{t^2}\\
= & e\sum_{t=1}^{+\infty}(\frac{e}{K})^K\frac{\left((2\log t)^2\log t\right)^K}{t^2}\\
= & e\sum_{t=1}^{+\infty}(\frac{e}{K})^K\frac{\left(4(\log t)^3\right)^K}{t^2}\\
= & e\sum_{t=1}^{+\infty}(\frac{4e (\log t)^3}{K})^K\frac{1}{t^2}\\
\end{align*}
$$


## Determine $\vec{w}_t$

+ If $\hat{\mu}_{a,t} < \mu_0$ holds for all $a$, we take $w_a^*=\frac{\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}{\sum_{i=1}^K\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}=\frac{\frac{1}{(\hat{\mu}_{a,t}-\mu_0)^2}}{\sum_{i=1}^K\frac{1}{(\hat{\mu}_{a,t}, \mu_0)^2}}$
+ If $\max_{1\leq a\leq K}\hat{\mu}_{a,t} > \mu_0$
  + If $i_t\in i^*(\hat{\mu_t})$, we take $\vec{w}_t=e_{i_t}$
  + If $i_t\notin i^*(\hat{\mu_t})$, $\vec{w}_t$ can be any vector in $\Delta_K$, as $\arg\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \neg i_t}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}=\arg\sup_{\vec{w}\in \Delta_K}0$. Here we take $\vec{w}_t=(\frac{1}{K},\frac{1}{K},\cdots,\frac{1}{K})$.

## The projection of $\vec{w}_t$​

After calculating the value of $\vec{w}_t$, we need to project it onto the class $\Sigma_K^{\epsilon_t}=\{(w_1,\cdots,w_k)\in [\epsilon_t, 1]^K: w_1+w_2+\cdots+w_K=1\}$, based on the $\infty$-norm. We need to derive the explicit expression of this formula.

Denote the projected $w$ onto $\Sigma_K^{\epsilon}$ as $\hat{w}$. 

+ Denote $S_1=\{i:w_i <\epsilon\}$, $S_2=\{i: w_i\geq \epsilon\}$

+ If $w_i < \epsilon$, we can assert $\hat{w}_i=\epsilon$.

  > Prove by contradiction, if $\exists i$, such that $\hat{w}_i > \epsilon$. 
  >
  > From the equality $\sum_{i\in S_1}(\hat{w}_i-w_i) + \sum_{i\in S_2}(\hat{w}_i-w_i)=0$, and the fact $\sum_{i\in S_1}(\hat{w}_i-w_i)\geq \sum_{i\in S_1}(\epsilon-w_i) > 0$, we know $\sum_{i\in S_2}(\hat{w}_i-w_i)<0$, further $\exists i'\in S_2$, such that $\hat{w}_{i'}-w_{i'}<0$.
  >
  > Take $\Delta = \min\{|\hat{w}_{i'}-w_{i'}|, |\hat{w}_i-\epsilon|\}$, and take $\tilde{w}_a=\begin{cases}\hat{w}_a & a\neq i,i'\\ \hat{w}_i'-\Delta & a=i\\ \hat{w}_i'+\Delta & a=i'\end{cases}$. Easy to see $\tilde{w}_a\geq \epsilon$, further $\tilde{w}\in \Sigma^{\epsilon}_K$.
  >
  > Then $|\tilde{w}_a-w_a|=|\hat{w}_a-w_a|$ for $a\neq i,i'$, and $\max\{|\tilde{w}_i-w_{i}|, |\tilde{w}_{i'}-w_{i'}|\}\leq \max\{|\hat{w}_i-w_{i}|, |\hat{w}_{i'}-w_{i'}|\}$. Thus, $\|\hat{w}-w\|_{\infty}\geq \|\tilde{w}-w\|_{\infty}$​.

+ Then it is suffice to find $\hat{w}_i, i\in S_2$, such that
  $$
  \sum_{i\in S_2} (w_i-\hat{w}_i ) = \sum_{i\in S_1}(\epsilon - w_i)\\
  \hat{w}_i\geq \epsilon
  $$

+ If $w_i \geq \epsilon$, we can assert $\hat{w}_i\leq w_i$​​.

  > Prove by contradiction. If $\exists i$, such that $w_i\geq \epsilon$, $\hat{w}_i > w_i$, then we can decrease $w_i$ a little bit while increasing some $\hat{w}_{i'}$ whose $w_{i'}-\hat{w}_{i'}<0$, then the infinity norm will be smaller.

+  



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

