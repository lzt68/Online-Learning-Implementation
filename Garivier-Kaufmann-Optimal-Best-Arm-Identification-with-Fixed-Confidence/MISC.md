# MISC





Linear Programming

When implementing C-Tracking, for a $\hat{w}$ based on the empirical mean rewards, we need to project $\hat{w}$ into $[0, 1]^K\cap \Sigma_K$. We conduct this project through solving the following linear optimization problem.
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
​      Formulate it as the matrix form, we have
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

```python
c = np.zeros(self.K + 1)
c[self.K] = 1

Aub = np.zeros((3 * self.K, self.K + 1))
Aub[0 : self.K, 0 : self.K] = -np.eye(self.K)
Aub[self.K : self.K * 2, 0 : self.K] = np.eye(self.K)
Aub[self.K : self.K * 2, self.K] = -1
Aub[self.K * 2 : self.K * 3, 0 : self.K] = -np.eye(self.K)
Aub[self.K * 2 : self.K * 3, self.K] = -1

bub = np.zeros(self.K * 3)
bub[0 : self.K] = -epsilon
bub[self.K : self.K * 2] = w
bub[self.K * 2 : self.K * 3] = -w

Aeq = np.ones((1, self.K + 1))
Aeq[0, self.K] = 0

beq = np.ones((1))
beq[0] = 1

res = linprog(c=c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq)
return res.x
```

