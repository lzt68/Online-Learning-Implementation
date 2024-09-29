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
   
   3. After calculating the value of $\vec{w}_t$, we need to project it onto the class $\Sigma_K^{\epsilon_t}=\{(w_1,\cdots,w_k)\in [\epsilon_t, 1]^K: w_1+w_2+\cdots+w_K=1\}$, based on the $\infty$-norm. We need to derive the explicit expression of this formula.
      
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