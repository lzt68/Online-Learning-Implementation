# README

This folder aims to apply the algorithm, in Degenne \& Koolen 2019 Pure Exploration with Multiple Correct Answers, to the *Any Low Problem*, stated as Example 1 in the paper. 

Paper Link: https://proceedings.neurips.cc/paper_files/paper/2019/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html

For convenience, here we consider **Any Large Problem**, which is essentially the same as the Example 1 in the paper. The problem formulation is as follows. For an instance $\nu$ with K arms, threshold $\mu_0$ (sometimes we use notation $\xi$ instead), and corresponding mean reward vector $\{r^{\nu}_{a}\}_{a=1}^K$​, we define "success" as

+ output an arm $a$ whose $r_a^{\nu}>\mu_0$, if $\max_{1\leq a\leq K}r_a^{\nu} > \mu_0$
+ output "none", if $\max_{1\leq a\leq K}r_a^{\nu} < \mu_0$

We want to design an algorithm which take confidence level $\delta$ as input, such that for any instance $\nu$, the algorithm can achieve success with probability $1-\delta$, while consuming as small pulling times as possible.

We only work on instances whose maximum mean reward is not $\mu_0$​​, and unit variance Gaussian Distribution.

## File Structure

+ "Source": the source file
  + agent.py: We implement three different versions of the algorithm Sticky-TaS, which are Sticky_TaS_old,  Sticky_TaS, Sticky_TaS_fast.
    The behavior of these implementations are the same, while the running efficiency of Sticky_TaS_fast is the best and Sticky_TaS_old is the worse.
    The main difference is how to figure out the first element in $I_t$. Sticky_TaS_fast implements an algorithm that can achieves $O(K\log K)$ time complexity in each round.
  + env.py: Define the Gaussian Instance.
  + Comparison-of-Three-Implementations.ipynb: This notebook compares the three implementations regarding their action and running speed. The numeric experiments show that given the same reward, these three implementations always adopt the same action. Regarding the execution speed, Sticky_TaS_fast is faster than Sticky-TaS. And Sticky-TaS is faster than Sticky_TaS_old. The difference of speed will become significant only when $K$ is large enough and $\delta$​ is small enough
  
+ Experiment.ipynb: This notebook conducts experiment on testing S-Tas

  1. The impact of the position of the qualified arm.
     It seems if the position of the qualified arm is 1, the stopping times can be much smaller.
  2. The convergence speed of the ration $\frac{\mathbb{E}\tau}{\log\frac{1}{\delta}}$.
     The convergence speed is very slow, as $\frac{\mathbb{E}\tau}{\log\frac{1}{\delta}}$ will close to $T^*$ only when $\delta< \exp(-100)$.

  

Following are the supportive proof for the implementation Sticky_TaS_fast.

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

In the following, we assume $\max_{1\leq a\leq K} \hat{\mu}_{i,t} \geq \mu_0$, $\sum_{i=1}^KN_i(t-1)\frac{(\hat{\mu}_{i,t}-\min\{\mu_0, \hat{\mu}_{i,t}\})^2}{2} \geq \log f(t-1)$ and only consider $a\neq \arg\max_{1\leq i\leq K} \hat{\mu}_{i,t}$.

> The time complexity to check $\max_{1\leq a\leq K} \hat{\mu}_{i,t} \geq \mu_0$ and $\sum_{i=1}^KN_i(t-1)\frac{(\hat{\mu}_{i,t}-\min\{\mu_0, \hat{\mu}_{i,t}\})^2}{2} \geq \log f(t-1)$ are both $\Theta(K)$.

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

Thus, we get $\inf_{\mu_a'\geq \mu_i', \forall i\neq a} \sum_{i=1}^K N_i(t-1)\frac{(\mu'_i-\hat{\mu}_{i,t})^2}{2}=N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i\neq a}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')=N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$. 



The remaining task is to solve $\min_{\mu_a' \geq \mu_0}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$. Denote 

+ sorted permutation of $\{\hat{\mu}_{i,t}\}_{i=1}^K$ as $i_1,i_2,\cdots,i_m$, such that $\hat{\mu}_{i_1, t} \geq \hat{\mu}_{i_2, t}\geq \cdots \geq \hat{\mu}_{i_m, t} > \hat{\mu}_{a,t}$.

+ $G(\mu_a')=N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{l=1}^m N_{i_l}(t-1)\frac{(\hat{\mu}_{i_l,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i_l,t} > \mu_a')$. We need to solve $\min_{\mu_a' \geq \mu_0} G(\mu_a') $.

+ $\mu_a^*=\arg\min_{\mu_a' \geq \mu_0}G(\mu_a')$, $\tilde{\mu}^*_a=\arg\min_{\mu_a'\geq \hat{\mu}_{a,t}} G(\mu_a')$
  Since $\arg\min_{\mu_a'\in\mathbb{R}} G(\mu_a') \in [\hat{\mu}_{a,t}, \hat{\mu}_{i_1, t}]$, we can conclude $\arg\min_{\mu_a'\in\mathbb{R}} G(\mu_a')=\tilde{\mu}^*$

  + If $\hat{\mu}_{a,t} > \mu_0$, we know $\mu_a^*=\tilde{\mu}^*_a$, as $\arg\min_{\mu_a' \geq \mu_0}G(\mu_a')\geq \arg\min_{\mu_a'\geq \hat{\mu}_{a,t}} G(\mu_a')$
  + If $\hat{\mu}_{a,t} \leq \mu_0$, we know $\mu_a^*=\max\{\mu_0, \tilde{\mu}^*_a\}$. The reason is function $G(\mu_a')$ is a convex function regarding $\mu_a'$. Then $G(\mu_a')$ is decreasing in the interval $(-\infty, \tilde{\mu}^*)$, while increasing in the interval $(\tilde{\mu}^*, +\infty)$.
    + If $\mu_0\leq \tilde{\mu}^*$, $\arg\min_{\mu_a' \geq \mu_0}G(\mu_a')$ is indeed the global minimum point of $G(\mu_a')$, which is $\tilde{\mu}^*$
    + If $\mu_0> \tilde{\mu}^*$, $G(\mu_a')$ is increasing in the interval $(\mu_0, +\infty)$. Then $\arg\min_{\mu_a' \geq \mu_0}G(\mu_a')=\mu_0$.

  That means to calculate $\mu_a^*$, suffice to calculate $\tilde{\mu}^*_a$, and then we can take $\mu_a^*=\max\{\tilde{\mu}_a^*, \mu_0\}$.

+ Denote ${c\mu}_j:=\frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j} N_{i_l}(t-1)}$ for all $j\in [m]$

We are going to prove $\tilde{\mu}_a^*={c\mu}_j$ for some $j\in [m]$.

> To get the minimum point of function $G(\mu_a')$ in the interval $[\hat{\mu}_{a,t}, \hat{\mu}_{i_1,t}]$, it is equivalent to find the minimum point in each small interval $[\hat{\mu}_{a,t}, \hat{\mu}_{i_m,t})$, $[\hat{\mu}_{i_m,t}, \hat{\mu}_{i_{m-1},t})$, ..., $[\hat{\mu}_{i_3,t}, \hat{\mu}_{i_2,t})$, $[\hat{\mu}_{i_2,t}, \hat{\mu}_{i_1,t}]$, and then take the point with smallest function value. By the fact that $G(\mu_a')$ is a convex function regarding $\mu_a'$ and is also smooth and convex in each small interval, we have the following conclusion 
> $$
> \begin{align*}
> & \tilde{\mu}_a^* =\arg\min_{\mu_a'\in[\hat{\mu}_{a,t}, \hat{\mu}_{i_1,t}]} N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')\\
> \Rightarrow & \exists j, \tilde{\mu}_a^*\in [\hat{\mu}_{i_{j}}, \hat{\mu}_{i_{j-1}}], \mu_a^* =\arg\min_{\mu_a'\in[\hat{\mu}_{i_{j}}, \hat{\mu}_{i_{j-1}}]} N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')\\
> \Leftrightarrow & \exists j, \tilde{\mu}_a^*\in [\hat{\mu}_{i_{j}}, \hat{\mu}_{i_{j-1}}], \mu_a^* =\arg\min_{\mu_a'\in[\hat{\mu}_{i_{j}}, \hat{\mu}_{i_{j-1}}]} N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{l=1}^{j-1} N_{i_l}(t-1)\frac{(\hat{\mu}_{i_l,t}-\mu_a')^2}{2}
> \end{align*}
> $$
> If $\exists j\in[m]$,  $\tilde{\mu}_a^* = \hat{\mu}_{i_{j},t}$, then we can conclude $\tilde{\mu}_a^* =\hat{\mu}_{i_{j},t}= \frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j-1} N_{i_l}(t-1)}$.
> 
> > Proof is as follows. Denote function $G_r(\mu_a')=N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{l=1}^{r} N_{i_l}(t-1)\frac{(\hat{\mu}_{i_l,t}-\mu_a')^2}{2}$ for $r\in[m]$. We have $G(\mu_a')=G_r(\mu_a')$ holds for all $\mu_a'\in[\hat{\mu}_{i_{r+1},t}, \hat{\mu}_{i_{r},t}]$ (we take $\hat{\mu}_{i_{m+1},t}=\hat{\mu}_{a,t}$)
> >
> > Not hard to see
> > $$
> > \frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{r} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{r} N_{i_l}(t-1)} = \arg\min_{\mu_a'\in \mathbb{R}} G_r(\mu_a'),
> > $$
> > holds for all $r\in [m]$. Since $\tilde{\mu}_a^* = \hat{\mu}_{i_{j},t}=\arg\min_{\mu_a' \geq \hat{\mu}_{a,t}}G(\mu_a')$, we can conclude
> > $$
> > G(\hat{\mu}_{i_{j},t})=G_j(\hat{\mu}_{i_{j},t})\leq \min_{\mu_a'\in[\hat{\mu}_{i_{j+1},t}, \hat{\mu}_{i_{j},t}]} G_{j}(\mu_a')\\
>> G(\hat{\mu}_{i_{j},t})=G_{j+1}(\hat{\mu}_{i_{j},t})\leq \min_{\mu_a'\in[\hat{\mu}_{i_{j+2},t}, \hat{\mu}_{i_{j+1},t}]} G_{j+1}(\mu_a')\\
> > $$
> > which means
> > $$
> > \hat{\mu}_{i_{j},t}=\arg\min_{\mu_a'\in[\hat{\mu}_{i_{j+1},t}, \hat{\mu}_{i_{j},t}]} G_{j}(\mu_a')=\arg\min_{\mu_a'\in[\hat{\mu}_{i_{j+2},t}, \hat{\mu}_{i_{j+1},t}]} G_{j+1}(\mu_a').
> > $$
> > Since $G_j, G_j+1$ are both quadratic functions, we can conclude
> > $$
> > \frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j} N_{i_l}(t-1)} \leq \hat{\mu}_{i_{j}},\\
> > \frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j+1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j+1} N_{i_l}(t-1)} \geq \hat{\mu}_{i_{j}}.
> > $$
> > Meanwhile, by the fact that $\hat{\mu}_{i_1, t} \geq \hat{\mu}_{i_2, t}\geq \cdots \geq \hat{\mu}_{i_m, t} > \hat{\mu}_{a,t}$, we have $\frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j} N_{i_l}(t-1)}\geq \frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j+1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j+1} N_{i_l}(t-1)}$, which implies
> > $$
> > \frac{N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^{j-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j-1} N_{i_l}(t-1)} = \hat{\mu}_{i_{j}}.
> > $$
> 
> If $\tilde{\mu}_a^*  \neq \hat{\mu}_{i_{j},t}$ forall $j$, by the convexity, we get
> $$
> \begin{align*}
> & \exists j, \mu_a^*\in [\hat{\mu}_{i_{j+1},t}, \hat{\mu}_{i_{j},t}], \tilde{\mu}_a^* =\arg\min_{\mu_a'\in[\hat{\mu}_{i_{j+1},t}, \hat{\mu}_{i_{j},t}]} N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{l=1}^{j} N_{i_l}(t-1)\frac{(\hat{\mu}_{i_l,t}-\mu_a')^2}{2}\\
> \Rightarrow & \exists j, \tilde{\mu}_a^* =\arg\min_{\mu_a'\in(\hat{\mu}_{i_{j+1},t}, \hat{\mu}_{i_{j},t})} N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{l=1}^{j} N_{i_l}(t-1)\frac{(\hat{\mu}_{i_l,t}-\mu_a')^2}{2}\\
> \stackrel{\text{Local optimality guarantess global optimality}}{\Rightarrow} & \exists j, \tilde{\mu}_a^* =\arg\min_{\mu_a'\in \mathbb{R}} N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{l=1}^{j} N_{i_l}(t-1)\frac{(\hat{\mu}_{i_l,t}-\mu_a')^2}{2}\\
> \Rightarrow &  \exists j,\tilde{\mu}_a^* =\frac{N_a(t-1)\mu_a' + \sum_{l=1}^{j} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j} N_{i_l}(t-1)}.
> \end{align*}
> $$
>In conclusion, we prove $\exists j, \tilde{\mu}_a^* =\frac{N_a(t-1)\mu_a' + \sum_{l=1}^{j} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j} N_{i_l}(t-1)}$. And there exists a $j$, such that
> $$
> \min_{\mu_a' \geq \mu_0}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')\\ = \left(N_a(t-1) + \sum_{l=1}^j N_{i_l}(t-1)\right)(\mu_a^*)^2 -2(\mu_a^*)\left(N_a(t-1)\hat{\mu}_{a,t} + \sum_{l=1}^j N_{i_l}(t-1)\hat{\mu}_{i_l, t}\right) + \left(N_a(t-1)(\hat{\mu}_{a,t})^2 + \sum_{l=1}^j N_{i_l}(t-1)(\hat{\mu}_{i_l, t})^2\right)
> $$

By the convexity of function $G(\mu_a')$, we know the optimal point is unique.

The algorithm for calculating the first element $i_t$ in each round $t$ is as follows. Given the empirical mean reward vector $\{\hat{\mu}_{a,t}\}_{a=1}^K$,

1. Sort $\{\hat{\mu}_{a,t}\}_{a=1}^K$ to determine $i_1,i_2,\cdots, i_K$, the time complexity is $O(K\log K)$.

2. Calculate $\sum_{l=1}^j N_{i_l}(t-1)$ for all $j$, $\sum_{l=1}^j N_{i_l}(t-1)\hat{\mu}_{i_l, t}$ for all $j$, $\sum_{l=1}^j N_{i_l}(t-1)(\hat{\mu}_{i_l, t})^2$ for all $j$. The time complexity is $O(K)$.

3. For $a=1,2,\cdots K$, use bisection search to calculate the optimal solution $\min_{\mu_a' \geq \mu_0}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$,

   We need to find an index $j\in [K]$ such that $\hat{\mu}_{i_{j}, t}\leq \frac{N_a(t-1)\mu_a' + \sum_{l=1}^{j-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j-1} N_{i_l}(t-1)}\leq \hat{\mu}_{i_{j-1}, t}$. Here $\frac{N_a(t-1)\mu_a' + \sum_{l=1}^{j-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{j-1} N_{i_l}(t-1)}$ increases as $j$​ decreases.

   + If $\hat{\mu}_{i_{r}, t}> \frac{N_a(t-1)\mu_a' + \sum_{l=1}^{r-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{r-1} N_{i_l}(t-1)}$​, 
     $$
     \begin{align*}
     & \hat{\mu}_{i_{r}, t}> \frac{N_a(t-1)\mu_a' + \sum_{l=1}^{r-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{r-1} N_{i_l}(t-1)}\\
     \Rightarrow & G_r(\mu_a')\text{ is increasing in the interval } [\hat{\mu}_{i_{r}, t}, \hat{\mu}_{i_{r-1}, t}]\\
     \Rightarrow & G(\mu_a') \text{ is increasing in the interval } [\hat{\mu}_{i_{r}, t}, \hat{\mu}_{i_{1}, t}]\\
     \Rightarrow & \text{We only needs to focus on index }j\leq r
     \end{align*}
     $$
    + If $\frac{N_a(t-1)\mu_a' + \sum_{l=1}^{r-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{r-1} N_{i_l}(t-1)}> \hat{\mu}_{i_{r-1}, t}$,
      $$
      \begin{align*}
      & \frac{N_a(t-1)\mu_a' + \sum_{l=1}^{r-1} N_{i_l}(t-1)\hat{\mu}_{i_l, t}}{N_a(t-1) + \sum_{l=1}^{r-1} N_{i_l}(t-1)}> \hat{\mu}_{i_{r-1}, t}\\
      \Rightarrow & G_r(\mu_a')\text{ is decreasing in the interval } [\hat{\mu}_{i_{r}, t}, \hat{\mu}_{i_{r-1}, t}]\\
      \Rightarrow & G(\mu_a')\text{ is decreasing in the interval } [\hat{\mu}_{a, t}, \hat{\mu}_{i_{r-1}, t}]\\
      \Rightarrow & \text{We only needs to focus on index }j\geq r
      \end{align*}
      $$
      
   
   This bisection search can be done with complexity $O(\log K)$​
   
   Once we complete the search, we only need $O(1)$ to calculate the $\min_{\mu_a' \geq \mu_0}N_a(t-1)\frac{(\hat{\mu}_{a,t}-\mu_a')^2}{2} + \sum_{i: \hat{\mu}_{i,t} > \hat{\mu}_{a,t}}N_i(t-1)\frac{(\hat{\mu}_{i,t}-\mu_a')^2}{2}\mathbb{1}(\hat{\mu}_{i,t} > \mu_a')$, by the pre-calculate value in step 2. And then compare it with $f(t)$

In total, The time complexity of this calculation is $\Theta(K\log K)$.  The time complexity of running the algorithm in each round is $\Theta(\log K)$. 

## Determine $i_t$

By the sticky pulling rule, we would select $i_t$ as the smallest arm index in the $I_t$. As the only usage of $I_t$ is to find out $i_t$ for the next step, there is no need to figure out all the elements in $I_t$. 

In the last section, we firstly calculate some values, whose time complexity is $\Theta(K\log K)$. Then we can iterate all the $a\in [K]$ and apply the bisection search, the total time complexity is $\Theta(K\log K)$. Then, in each round, we need $\Theta(K\log K)$ calculations to determine $i_t$.

## Specify the value of Constant $C$
In the Theorem 10, the authors require to take $\beta(t,\delta)=\log\frac{Ct^2}{\delta}$ with $C\geq e\sum_{t=1}^{+\infty}(\frac{e}{K})^K\frac{(\left(\log (Ct^2))^2\log t\right)^K}{t^2}$, without delivering an exact value of $C$. This constant also occurs at the Lemma 14 at page 7, which take $f(t)=Ct^{10}$​.

It is unclear how to specify the value of $C$. Notice that if we take $C\geq 1$​, we have
$$
\begin{align*}
& e\sum_{t=1}^{+\infty}(\frac{e}{K})^K\frac{\left(\log (Ct^2))^2\log t\right)^K}{t^2}\\
\geq & e(\frac{e}{K})^K\frac{\left(\log (Ct^2))^2\log t\right)^K}{t^2}|_{t=\lceil e^K\rceil}\\
\geq & e(\frac{e}{K})^K \frac{(\log (C e^{4K}))^2 K)^K}{e^{2K}}\\
\stackrel{C\geq 1}{\geq} & e(\frac{e}{K})^K \frac{(16K^2 K)^K}{(e^{K}+1)^2}\\
= & e(\frac{e}{K})^K \frac{16^ K K^{3K}}{(e^{K}+1)^2}= e \frac{16^ K K^{2K}}{2e^{K}}.
\end{align*}
$$
That means $C$ cannot be independent of the arm number $K$, from the perspective of theoretical analysis.

## Determine $\vec{w}_t$

+ If $\hat{\mu}_{a,t} < \mu_0$ holds for all $a$, we take $w_a^*=\frac{\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}{\sum_{i=1}^K\frac{1}{d(\hat{\mu}_{a,t}, \mu_0)}}=\frac{\frac{1}{(\hat{\mu}_{a,t}-\mu_0)^2}}{\sum_{i=1}^K\frac{1}{(\hat{\mu}_{a,t}, \mu_0)^2}}$
+ If $\max_{1\leq a\leq K}\hat{\mu}_{a,t} > \mu_0$
  + If $i_t\in i^*(\hat{\mu}_t)$, we take $\vec{w}_t=e_{i_t}$
  + If $i_t\notin i^*(\hat{\mu}_t)$, $\vec{w}_t$ can be any vector in $\Delta_K$, as $\arg\sup_{\vec{w}\in \Delta_K}\inf_{\vec{\lambda}\in \neg i_t}\sum_{a=1}^K w_a\frac{(\mu_a-\lambda_a)^2}{2}\stackrel{\vec{\lambda}=\vec{\mu}}{=}\arg\sup_{\vec{w}\in \Delta_K}0$. Here we take $\vec{w}_t=(\frac{1}{K},\frac{1}{K},\cdots,\frac{1}{K})$.

## The projection of $\vec{w}_t$​

After calculating the value of $\vec{w}_t$, we need to project it onto the class $\Sigma_K^{\epsilon_t}=\{(w_1,\cdots,w_k)\in [\epsilon_t, 1]^K: w_1+w_2+\cdots+w_K=1\}$, based on the $\infty$-norm. We need to derive the explicit expression of this formula.

> It is not hard to see there could be multiple $w'\in \Sigma_K^{\epsilon}$ such that minimize $\|w-w'\|_{\infty}$. For example, take $w=(\frac{2}{9},\frac{2}{9},\frac{2}{9},\frac{1}{3}, 0)$, $\epsilon = \frac{1}{6}$, then both $(\frac{2}{9},\frac{2}{9},\frac{2}{9},\frac{1}{6}, \frac{1}{6})$ and $(\frac{2}{9}-\frac{1}{24},\frac{2}{9}-\frac{1}{24},\frac{2}{9}-\frac{1}{24},\frac{1}{3}-\frac{1}{24}, \frac{1}{6})$ would achieve minimum infinity norm value $\frac{1}{6}$. It seems for the algorithm, there isn't any difference between adopting various projections.
>
> In the following, we aim we to find one possible projections.

If all the entries of $w$ is above $\epsilon$, then $w$ itself is the projection. Here we only focus on the case the minimum entry of $w$ is below $\epsilon$.

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

  That means $\min_{w'\in \Sigma_K^{\epsilon }}\|w'-w\|_{\infty}=\min\limits_{w'\in \Sigma_K^{\epsilon };\forall i\in S_1 w'_i=\epsilon}\|w'-w\|_{\infty}$.

+ Then it is suffice to find $\hat{w}_i, i\in S_2$, such that
  $$
  \sum_{i\in S_2} (w_i-\hat{w}_i ) = \sum_{i\in S_1}(\epsilon - w_i)\\
  \hat{w}_i\geq \epsilon
  $$

+ If $w_i \geq \epsilon$, we can assert $\hat{w}_i\leq w_i$​​.

  > Prove by contradiction. If $\exists i$, such that $w_i\geq \epsilon$, $\hat{w}_i > w_i$, then we can decrease $w_i$ a little bit while increasing some $\hat{w}_{i'}$ whose $w_{i'}-\hat{w}_{i'}<0$, then the infinity norm will be smaller.

  That means $\min_{w'\in \Sigma_K^{\epsilon }}\|w'-w\|_{\infty}=\min\limits_{w'\in \Sigma_K^{\epsilon };\forall i\in S_1 w_i'=\epsilon; \forall i\in S_2, w_i' \leq w_i}\|w'-w\|_{\infty}$.

+ An algorithm for determining $\hat{w}_i, i\in S_2$. Assume $|S_2|=m$.

  1. Sort the elements in $S_2$, denote $i_1,i_2,\cdots, i_m \in S_2$, such that $\epsilon \leq w_{i_1} \leq w_{i_2}\leq \cdots\leq w_{i_m}$. Denote $B = \sum_{i\in S_1}(\epsilon - w_i)$.
  2. For $j=1,2,\cdots, m$
     + If $w_{i_j}-\epsilon  > \frac{B}{m-j+1}$, then take $w'_{i_l}=w_{i_l} - \frac{B}{m-j+1}, l=j,j+1,\cdots, m$, break the loop.
     + If $w_{i_j}-\epsilon \leq \frac{B}{m-j+1}$, then take $w'_{i_j}=\epsilon$, $B=B-(w_{i_j}-\epsilon)$. Go the next loop for $j+1$

  Easy to see $w_{i_j}-\epsilon \leq \frac{B}{m-j+1}\Leftrightarrow \frac{B-(w_{i_j}-\epsilon)}{m-j} \geq \frac{B}{m-j+1}$
  By applying this loop, assume at $j_0$, we have
  $$
  w_{i_1}'=\epsilon, \cdots ,w_{i_{j_0}}'=\epsilon\\
  w_{i_{j_0+1}}'=w_{i_{j_0+1}}-\frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}, w_{i_{j_0+2}}'=w_{i_{j_0+2}}-\frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0},\cdots w_{i_{m}}'=w_{i_{m}}-\frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}
  $$
  Then, we prove the constructed $w'$ can indeed achieve the minimum infinity norm. Easy to see
  $$
  \|w-w'\|_{\infty} =\max\left\{\max_{i\in S_1}\epsilon-w_i, w_{j_0}-\epsilon, \frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}\right\}
  $$

  + If $\max\left\{\max_{i\in S_1}\epsilon-w_i, w_{j_0}-\epsilon, \frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}\right\} = \max_{i\in S_1}\epsilon-w_i$, then we are done. As $\|w-w'\|_{\infty}\geq \max_{i\in S_1}\epsilon-w_i$ for $w'\in\Sigma_K^{\epsilon}$.
    Then we only need to focus on the case $\max_{i\in S_1}\epsilon-w_i < \max\left\{ w_{j_0}-\epsilon, \frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}\right\}$.

  + From the algorithm, we know
    $$
    w_{j_0}-\epsilon\leq \frac{B-\sum_{j=1}^{j_0-1}(w_{i_j}-\epsilon)}{m-j_0+1} \leq \frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0},
    $$
    which implies $\|w-w'\|_{\infty} = \max\left\{ w_{j_0}-\epsilon, \frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}\right\} = \frac{B-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}$. 

    

As we have shown $\min_{w'\in \Sigma_K^{\epsilon }}\|w'-w\|_{\infty} = \min\limits_{w'\in \Sigma_K^{\epsilon };\forall i\in S_1 w_i'=\epsilon; \forall i\in S_2, w_i' \leq w_i}\|w'-w\|_{\infty}$. The remaining task is to show $\min\limits_{w'\in \Sigma_K^{\epsilon };\forall i\in S_1 w_i'=\epsilon; \forall i\in S_2, w_i' \leq w_i}\|w'-w\|_{\infty}\geq \frac{\sum_{i\in S_1}(\epsilon - w_i)-\sum_{j=1}^{j_0}(w_{i_j}-\epsilon)}{m-j_0}$.
$$
\begin{align*}
& 1=\sum_{i=1}^K w_i = \sum_{i=1}^K w_i'\\
\Rightarrow & 0 = \sum_{i \in  S_1} (w_i'-w_i) + \sum_{j=1}^{m} (w_{i_j}'-w_{i_j})\\
\Rightarrow & 0 \geq \sum_{i \in  S_1} (\epsilon-w_i) + \sum_{j=1}^{j_0} (\epsilon-w_{i_j}) + \sum_{j=j_0+1}^{m} (w_{i_j}'-w_{i_j})\\
\Leftrightarrow & \sum_{j=j_0+1}^{m} (w_{i_j}-w_{i_j}') \geq \sum_{i \in  S_1} (\epsilon-w_i) + \sum_{j=1}^{j_0} (w_{i_j}'-w_{i_j})\\
\Rightarrow & \min_{j=j_0+1,\cdots, m} w_{i_j}-w_{i_j}' \geq \frac{\sum_{i \in  S_1} (\epsilon-w_i) + \sum_{j=1}^{j_0} (w_{i_j}'-w_{i_j})}{m-j_0}\\
\Rightarrow & \min\limits_{w'\in \Sigma_K^{\epsilon };\forall i\in S_1 w_i'=\epsilon; \forall i\in S_2, w_i' \leq w_i}\|w'-w\|_{\infty} \geq \frac{\sum_{i \in  S_1} (\epsilon-w_i) + \sum_{j=1}^{j_0} (w_{i_j}'-w_{i_j})}{m-j_0}
\end{align*}
$$
Then we have found a way to derive the explicit projection of vector $w$.

## Determine whether to stop

The stopping rule in the algorithm is $\exist i\in [K]\cup\{\text{none}\}$, such that $\{\vec{\mu'}: D(N_t,\hat{\mu}_t,\vec{\mu'})\leq \log\frac{Ct^2}{\delta}\} \cap \neg i=\emptyset$​.

A brute-force way is to iterate all the arms $i\in [K]\cup\{\text{none}\}$, and then check whether $\min_{\vec{\mu'}\in \neg i} D(N_t,\hat{\mu}_t,\vec{\mu'}) > \log\frac{Ct^2}{\delta}$ holds, as $\min_{\vec{\mu'}\in \neg i} D(N_t,\hat{\mu}_t,\vec{\mu'}) > \log\frac{Ct^2}{\delta}\Leftrightarrow \{\vec{\mu'}: D(N_t,\hat{\mu}_t,\vec{\mu'})\leq \log\frac{Ct^2}{\delta}\} \cap \neg i=\emptyset$.
For arm $i=1,2,\cdots, K$

+ If $\hat{\mu}_{i,t} > \mu_0$, we can take $\vec{\mu'}_a=\begin{cases}\mu_0 & a=i\\ \hat{\mu}_{a,t} & a\neq i\end{cases}$ , that means if $N_{i}(t-1)\frac{(\hat{\mu}_{i, t}-\mu_0)^2}{2} > \log\frac{Ct^2}{\delta}$ holds, we can output arm $i$
+ If $\hat{\mu}_{i,t} \leq \mu_0$, we know $\hat{\mu}_t \in \neg i$, further $\min_{\vec{\mu'}\in \neg i} D(N_t,\hat{\mu}_t,\vec{\mu'})=0 < \log\frac{Ct^2}{\delta}$. That means we would not output i.

For arm $i=\text{none}$,

+ If $\max_a \hat{\mu}_{a,t} \geq \mu_0$, then $\hat{\mu}_t\in \neg\text{none}$, further $\min_{\vec{\mu'}\in \neg \text{none}} D(N_t,\hat{\mu}_t,\vec{\mu'})=0 < \log\frac{Ct^2}{\delta}$. That means we would not output $\text{none}$.
+ If $\max_a \hat{\mu}_{a,t} < \mu_0$, then $\min_{\vec{\mu'}\in \neg \text{none}} D(N_t,\hat{\mu}_t,\vec{\mu'})=\min_{a\in [K]} N_t \frac{(\hat{\mu}_{a,t}-\mu_0)^2}{2}$. If we have $\min_{a\in [K]} N_t \frac{(\hat{\mu}_{a,t}-\mu_0)^2}{2} > \log\frac{Ct^2}{\delta}$, we can output $\text{none}$.



