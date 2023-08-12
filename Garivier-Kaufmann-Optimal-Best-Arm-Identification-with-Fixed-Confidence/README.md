# README

This folder aims to implement the **Track-and-Stop** algorithms in *Garivier \& Kaufamm Optimal Best Arm Identification with Fixed Confidence* . Link of the paper:http://arxiv.org/abs/1602.04589. 

There are two versions of tracking strategy mentioned in the paper, which are C-Tracking and D-Tracking. Though the paper compared their proposed algorithms with others as numeric record, here I don't follow their ideas. Instead, I am going to test the changing trend of $\frac{\mathbb{E}_{\mu}[\tau_{\delta}]}{\log\frac{1}{\delta}}$, which is also the main theoretical result in the paper.

Here we assume the reward follows Bernoulli Distribution.

## Remarks

1. When we calculate $x\log \frac{x}{y}+(1-x)\log\frac{1-x}{1-y}$, we set a minimum threshold 0.001 and maximum threshold 0.999 for both x and y to avoid numeric error.

2. 
   When we calculate the optimal pulling fraction using empirical mean reward, it is possible $\hat{\mu}_1=\hat{\mu}_a$. Recall the proposed algorithm in the paper cannot deal with the case of multiple optimal arms. And the Theorem 1, Lemma 3, Lemma 4, Theorem 5 fail to apply to this case. In the implementation, we consider "unique optimal arm" is prior knowledge, and we use following limit to replace the ratio in the definition of $F_{\mu}$ in Theorem 5.
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
   
   3. Define $S_1=\{i: \hat{w}_i < \epsilon\}$, $S_2=\{i: \hat{w}_i \geq \epsilon\}$. If $S_1=\emptyset$, then projected $w$ is $\hat{w}$

      Define projected $w$ as
      $$
      w_i =\begin{cases}
        \epsilon &  i\in S_1\\
        \hat{w}_i-\frac{\sum_{j\in S_1}\epsilon-\hat{w}_j}{|S_2|} & i\in S_2
        \end{cases}
      $$
      Prove $w = \arg\min_{\tilde{w}\in \Sigma_{K}\cap [\epsilon, 1]^K} \|\tilde{w}-\hat{w}\|_{\infty}$, as the projected vector.

      By our definition
      $$
      \|w - \hat{w}\|_{\infty} = \max\left\{\max_{i\in S_1}\epsilon-\hat{w}_i, \frac{\sum_{j\in S_1}\epsilon-\hat{w}_j}{|S_2|} \right\}
      $$
      Easy to see  $\forall \tilde{w}\in \Sigma_{K}\cap [\epsilon, 1]^K,  \|\tilde{w}-\hat{w}\|_{\infty}\geq \max_{i\in S_1}\epsilon-\hat{w}_i$, Then suffices to prove $\forall \tilde{w}\in \Sigma_{K}\cap [\epsilon, 1]^K, \|\tilde{w}-\hat{w}\|_{\infty}\geq \frac{\sum_{j\in S_1}\epsilon-\hat{w}_j}{|S_2|}$.

      For all $w\in \Sigma_{K}\cap [\epsilon, 1]^K$, we have
      $$
        \begin{align*}
        &\sum_{i\in S_1}\hat{w}_i+\sum_{i\in S_2}\hat{w}_i = \sum_{i\in S_1}w_i+\sum_{i\in S_2}w_i=1\\
        \Rightarrow & \left(\sum_{i\in S_1}\epsilon-\hat{w}_i\right)-\sum_{i\in S_2}\hat{w}_i + \sum_{i\in S_1}w_i+\sum_{i\in S_2}w_i=|S_1|\epsilon\\
        \Leftrightarrow & \sum_{i\in S_2}(w_i-\hat{w}_i) = |S_1|\epsilon - \sum_{i\in S_1}w_i - \left(\sum_{i\in S_1}\epsilon-\hat{w}_i\right)
        \end{align*}
      $$
      Since $w_i\geq \epsilon$, thus $|S_1|\epsilon - \sum_{i\in S_1}w_i\leq 0, |S_1|\epsilon - \sum_{i\in S_1}w_i - \left(\sum_{i\in S_1}\epsilon-\hat{w}_i\right) < 0$, we have
      $$
        \begin{align*}
        & \sum_{i\in S_2}(w_i-\hat{w}_i) \leq  - \left(\sum_{i\in S_1}\epsilon-\hat{w}_i\right)\\
        \Rightarrow & \min_{i\in S_2} w_i-\hat{w}_i\leq - \frac{\sum_{i\in S_1}\epsilon-\hat{w}_i}{|S_2|}\\
        \Rightarrow & \max_{i\in S_2} \hat{w}_i-w_i\geq  \frac{\sum_{i\in S_1}\epsilon-\hat{w}_i}{|S_2|}\\
        \Rightarrow & \max_{i\in S_2} |\hat{w}_i-w_i|\geq \frac{\sum_{i\in S_1}\epsilon-\hat{w}_i}{|S_2|}\\
        \Rightarrow & \max_{i\in [K]} |\hat{w}_i-w_i|\geq \frac{\sum_{i\in S_1}\epsilon-\hat{w}_i}{|S_2|}\\
        \end{align*}
      $$

      which implies $\|w-\hat{w}\|_{\infty}\geq \frac{\sum_{i\in S_1}\epsilon-\hat{w}_i}{|S_2|}$. Combining with $\|w-\hat{w}\|_{\infty}\geq \max_{i\in S_1}\epsilon-\hat{w}_i$, we have
      $$
      \|w - \hat{w}\|_{\infty} \geq \max\left\{\max_{i\in S_1}\epsilon-\hat{w}_i, \frac{\sum_{j\in S_1}\epsilon-\hat{w}_j}{|S_2|} \right\},
      $$
      which implies our definition is indeed the projection.