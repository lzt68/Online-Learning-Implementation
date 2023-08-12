# README

This folder aims to implement the **Track-and-Stop** algorithms in *Garivier \& Kaufamm Optimal Best Arm Identification with Fixed Confidence* . Link of the paper:http://arxiv.org/abs/1602.04589. 

There are two versions of tracking strategy mentioned in the paper, which are C-Tracking and D-Tracking. Though the paper compared their proposed algorithms with others as numeric record, here I don't follow their ideas. Instead, I am going to test the changing trend of $\frac{\mathbb{E}_{\mu}[\tau_{\delta}]}{\log\frac{1}{\delta}}$, which is also the main theoretical result in the paper.

Here we assume the reward follows Bernoulli Distribution.

## Remarks

1. When we calculate $x\log \frac{x}{y}+(1-x)\log\frac{1-x}{1-y}$, we set a minimum threshold 0.001 and maximum threshold 0.999 for both x and y to avoid numeric error.

2. When we calculate the optimal pulling fraction using empirical mean reward, it is possible $\hat{\mu}_1=\hat{\mu}_a$. Recall the proposed algorithm in the paper cannot deal with the case of multiple optimal arms. And the Theorem 1, Lemma 3, Lemma 4, Theorem 5 fail to apply to this case. In the implementation, we consider "unique optimal arm" is prior knowledge, and we use following limit to replace the ratio in the definition of $F_{\mu}$ in Theorem 5.
   $$
   \begin{align*}
   &\lim_{\mu_a \rightarrow \mu_1^-} \frac{\mu_1\log\frac{\mu_1}{w_1\mu_1+w_a\mu_a}+(1-\mu_1)\log\frac{1-\mu_1}{1-w_1\mu_1-w_a\mu_a}}{\mu_a\log\frac{\mu_a}{w_1\mu_1+w_a\mu_a}+(1-\mu_a)\log\frac{1-\mu_a}{1-w_1\mu_1-w_a\mu_a}}\\
   =&\lim_{\mu_a \rightarrow \mu_1^-} \frac{-\mu_1\frac{w_a}{w_1\mu_1+w_a\mu_a}-(1-\mu_1)\frac{-w_a}{1-w_1\mu_1-w_a\mu_a}}{\left(1+\log \mu_a\right)-\left(\frac{w_a\mu_a}{w_1\mu_1+w_a\mu_a}+\log(w_1\mu_1+w_a\mu_a) \right)-\left(1+\log (1-\mu_a)\right)-\left(\frac{-w_a(1-\mu_a)}{1-w_1\mu_1-w_a\mu_a}-\log (1-w_1\mu_1-w_a\mu_a) \right)}\\
   =&\lim_{\mu_a \rightarrow \mu_1^-} \frac{\frac{\mu_1 w_a^2}{(w_1\mu_1+w_a\mu_a)^2}+\frac{(1-\mu_1)w_a^2}{(1-w_1\mu_1-w_a\mu_a)^2}}{\frac{1}{\mu_a}-\left(\frac{w_1w_a\mu_1}{(w_1\mu_1+w_a\mu_a)^2}+\frac{w_a}{w_1\mu_1+w_a\mu_a}\right)-\left(-\frac{1}{1-\mu_a}\right)-\left(-\frac{w_1(1-\mu_1)(-w_a)}{[w_a(1-\mu_a)+w_1(1-\mu_1)]^2}-\frac{-w_a}{1-w_1\mu_1-w_a\mu_a}\right)}\\
   =&\frac{\frac{w_a^2}{\mu_1(1-\mu_1)}}{\frac{1}{\mu_1}-\left(\frac{w_1w_a}{\mu_1}+\frac{w_a}{\mu_1}\right)+\left(\frac{1}{1-\mu_1}\right)-\left(\frac{w_1w_a}{1-\mu_1}+\frac{w_a}{1-\mu_1}\right)}\\
   =&\frac{\frac{w_a^2}{\mu_1(1-\mu_1)}}{\frac{1}{\mu_1}-\frac{w_1w_a}{\mu_1}-\frac{w_a}{\mu_1}+\left(\frac{1}{1-\mu_1}\right)-\left(\frac{w_1w_a}{1-\mu_1}+\frac{w_a}{1-\mu_1}\right)}\\
   =&\frac{w_a^2}{w_1-w_1w_a}\\
   =&\frac{w_a^2}{w_1^2}
   \end{align*}
   $$
   Here $w_1=\frac{1}{1+x_a(y)}, w_a=\frac{x_a(y)}{1+x_a(y)}$.

   3. When implementing C-Tracking, for a $\hat{w}$ based on the empirical mean rewards, we need to project $\hat{w}$ into $[0, 1]^K\cap \Sigma_K$. We conduct this project through solving the following linear optimization problem.
      $$
      \begin{array}{rl}
      \min & t\\
      s.t. & w_i\geq \epsilon, \forall i\\
      & w_i - \hat{w}_i\leq t\\
      & -w_i + \hat{w}_i\leq t\\
      & \sum_{i=1}^K w_i=1\\
      & t\geq 0
      \end{array}
      $$
      Formulate it as the matrix form, we have
      $$
      \begin{array}{rl}
      \min & [\underbrace{0, \cdots, 0}_K, 1][w_1, \cdots, w_K, t]^T\\
      s.t. & 
      \left[\begin{matrix}
      -I_K & 0 \\
      I_k & -1 \\
      -I_k & -1 \\
      \end{matrix}\right][w_1, \cdots, w_K, t]^T \leq 
      [\underbrace{-\epsilon, \cdots, -\epsilon}_{K}, \hat{w}_1, \cdots, \hat{w}_K, -\hat{w}_1, \cdots, -\hat{w}_K]\\
      & [\vec{1}_K \ 0] [w_1, \cdots, w_K, t]^T = 1
      \end{array}
      $$
      