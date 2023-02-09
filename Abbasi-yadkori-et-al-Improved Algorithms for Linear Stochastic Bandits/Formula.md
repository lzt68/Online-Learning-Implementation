$\|x\|_A:=\sqrt{x^TAx}$, $<x,y>_A:=x^TAy$, define $\bar{V}_t:=V+\sum_{s=1}^t X_sX_s^T$, $S_t:=\sum_{s=1}^{t} \eta_s X_s$

$Y_t=<X_t, \theta_*>+\eta_t$, 

$X_{1:t}:=\left[\begin{array}{c}X_1^T\\\vdots\\ X_t^T\end{array}\right]$, $Y_{1:t}:=\left[\begin{array}{c}Y_1\\\vdots\\ Y_t\end{array}\right]$

$\hat{\theta}_t=(X^T_{1:t}X_{1:t}+\lambda I)^{-1} X^T_{1:t}Y_{1:t}$

$\|\theta_*\|_2 \le S$

Assume V is $d\times d$ positive definite matrix. 



Let $\{F_t\}_{t=0}^{+\infty}$ be a filtration, $\{\eta_t\}_{t=1}^{+\infty}$ be a real-valued stochastic process such that $\eta_t$ is $F_t$-measurable and $\eta_t$ satisfy $\forall R, \mathbb{E}[e^{\lambda \eta_t}|F_{t-1}]\le \exp\left(\frac{\lambda^2 R^2}{2}\right)$. Let $\{X_t\}_{t=1}^{\infty}$ be an $\mathbb{R}^d$-valued stochastic process such that $X_t$ is $F_{t-1}$-measurable. For any $t\ge 0$,  Then for $\delta>0$, with probability at least $1-\delta$, for all $t\ge 0$, $\|S_t\|^2_{\bar{V}^{-1}_t}\le 2R^2\log\left(\frac{\det(\bar{V}_t)^{\frac{1}{2}}\det\left(V\right)^{\frac{1}{2}}}{\delta}\right)$

Assume the same as in Theorem 1, let $V=\lambda I$, $\lambda >0$. Then for any $\delta>0$, with probability at least $1-\delta$, for all $t\ge 0 $, $\theta_*$ lies in the set 
$$
C_t=\left\{\theta\in \mathbb{R}^d:\|\hat{\theta}_t-\theta\|_{\bar{V}_t}\le R\sqrt{2\log\left(\frac{\det(\bar{V})^{\frac{1}{2}}\det(V)^{\frac{1}{2}}}{\delta}\right)}+\lambda^{\frac{1}{2}}S\right\}
$$
Furthermore, if for all $t\ge 1$, $\|x_t\|_2\le 1$, then with prob at least $1-\delta$, for all $t\ge 0$, $\theta_*$ lies in the set
$$
C'_t=\left\{\theta\in\mathbb{R}^d: \|\hat{\theta}_t-\theta\|_{\bar{V}_t}\le R\sqrt{d\log\left(\frac{1+tL^2/\lambda}{\delta}\right)}+\lambda^{\frac{1}{2}}S\right\}
$$


Assume that for all $t$ and $x\in D_t$, $<x,\theta_{*}>\in [-1, 1]$. Then, with probability at least $1-\delta$, the regret of the OFUL algorithm satisfies
$$
\forall n\ge 0, R_n\le 4\sqrt{nd \log\left(\lambda+nL/d\right)}\left(\lambda^{\frac{1}{2}}S+R\sqrt{2\log\frac{1}{\delta}+d\log(1+\frac{nL}{\lambda d})}\right)
$$
Let $\lambda\in\mathbb{R}^d$ be arbitrary and consider for any $t\ge 0$, $M_t^{\lambda}:=\exp\left(\sum_{s=1}^t\left[\frac{\eta_s<\lambda,X_s>}{R}-\frac{1}{2}<\lambda,X_s>^2\right]\right)$. Let $\tau$ be a stopping time with respect to the filtration $\{F_t\}_{t=0}^{+ \infty}$. Then $M^{\lambda}_{\tau}$ is almost surely well defined and $\mathbb{E}M_\tau^{\lambda}\le 1$



Let $\tau$ be a stopping time with respect to the filtration $\{F_t\}_{t=0}^{+\infty }$. Then, for $\delta>0$, with probability $1-\delta$, $\|S_t\|^2_{\bar{V}^{-1}_\tau}\le 2R^2\log\left(\frac{\det(\bar{V}_{\tau})^{\frac{1}{2}}\det\left(V\right)^{\frac{1}{2}}}{\delta}\right)$



Suppose $X_1,X_2,\cdots,X_t\in \mathbb{R}^d$ and for any $1\le s\le t$, $\|X_S\|_2\le L$. Let $\bar{V}_t=\lambda I+\sum_{s=1}^t X_sX_s^T$ for some $\lambda>0$, then $\det(\bar{V}_t)\le (\lambda+\frac{tL^2}{d})^d$



Let $\{X_t\}_{t=1}^{+\infty}$ be a sequence in $\mathbb{R}^d$, $V$ a $d \times d$ positive define matrix and define $\bar{V}_t=V+\sum_{s=1}^{t}X_sX_s^T$, then we have $\log\left(\frac{\det(\bar{V}_n)}{\det(V)}\right)\le \sum_{t=1}^n \|X_t\|_{\bar{V}_{t-1}^{-1}}^2$. Further, if $\|X_t\|_2\le L$ for all $t$, then
$$
\begin{align*}
\sum_{t=1}^n\min\{1, \|X_t\|^2_{\bar{V}_{t-1}^{-1}}\}\le &2(\log \det(\bar{V}_n)-\log\det V)\\
\le & 2\left(d\log\left(\frac{trace(V)+nL^2}{d}\right)-\log\det V\right)
\end{align*}
$$
and finally, if $\lambda_{\min}(V)\ge \max\{1, L^2\}$ then
$$
\sum_{t=1}^n \|X_t\|_{\bar{V}_{t-1}^{-1}}^2\le 2\log\frac{\det\bar{V}_n}{\det V}
$$


# Find the point with maximum l2 norm on the ellipsoid

$$
\begin{array}{cc}
\max & x^Tx \\
s.t. & (x-x_0)^T A(x-x_0)\le\beta
\end{array}
$$

where $x_0$, $A$, $\beta$ are known to us.

Define $L(x, \lambda) = x^Tx - \lambda(\beta-(x-x_0)^T A(x-x_0))$, then $\frac{\partial L}{\partial x}=0, \frac{\partial L}{\partial \lambda}=0$ are equivalent to
$$
\begin{align*}
2x+2\lambda A(x-x_0)=&0\\
\beta-(x-x_0)^T A(x-x_0)=&0
\end{align*}
$$

$$
\begin{align*}
2x+2\lambda Ax -2\lambda Ax_0=&0\\
\beta=&x^Tx-2x_0^TA x+x_0^Tx_0
\end{align*}
$$

Let $x=x^{(0)}+\theta_x$, $\lambda=\lambda^{(0)}+\theta_{\lambda}$, we have
$$
\bigg\{
\begin{align*}
(x^{(0)}+\theta_x)+(\lambda^{(0)}+\theta_{\lambda}) A(x^{(0)}+\theta_x) -(\lambda^{(0)}+\theta_{\lambda}) Ax_0=&0\\
(x^{(0)}+\theta_x)^T(x^{(0)}+\theta_x)-2x_0^TA(x^{(0)}+\theta_x)+x_0^Tx_0=\beta
\end{align*}
$$

$$
\begin{align*}
&(x^{(0)}+\theta_x)+(\lambda^{(0)}+\theta_{\lambda}) A(x^{(0)}+\theta_x) -(\lambda^{(0)}+\theta_{\lambda}) Ax_0=0\\
\Rightarrow&x^{(0)}+\theta_x+\lambda^{(0)}Ax^{(0)}+\lambda^{(0)}A\theta_x+\theta_{\lambda}Ax^{(0)}+\theta_{\lambda}A\theta_x-\lambda^{(0)}Ax_0-\theta_{\lambda}Ax_0=0\\
\Rightarrow& (I+\lambda^{(0)}A)\theta_x+(Ax^{(0)}-Ax_0)\theta_{\lambda}+(\theta_{\lambda}A\theta_x)=-x^{(0)}-\lambda^{(0)}Ax^{(0)}+\lambda^{(0)}Ax_0
\end{align*}
$$

$$
\begin{align*}
&(x^{(0)}+\theta_x)^T(x^{(0)}+\theta_x)-2x_0^TA(x^{(0)}+\theta_x)+x_0^Tx_0=\beta\\
\Rightarrow& x^{(0)T}x^{(0)}+2x^{(0)T}\theta_x+\theta_x^T\theta_x-2x_0^TAx^{(0)}-2x_0^TA\theta_x+x_0^Tx_0=\beta\\
\Rightarrow&(2x^{(0)T}-2x_0^TA)\theta_x+(\theta_x^T\theta_x)=\beta-x^{(0)T}x^{(0)}+2x_0^TAx^{(0)}-x_0^Tx_0
\end{align*}
$$

