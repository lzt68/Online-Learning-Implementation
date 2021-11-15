# Gradient of neural network
Algorithm of Neural Network
$$
\begin{align}
X_0 &= X\\
X_1 &=\sigma(W_1X_0)\\
X_2 &=\sigma(W_2X_1)\\
\cdots\\
X_{L-1}&=\sigma(W_{L-1}X_{L-2})\\
X_{L} &=\sqrt{m} * W_L X_{L-1}
\end{align}
$$
$f(X) = X_L$

Backward Propagation
$$
\begin{align}
\nabla_{X_L}f &= 1\\
\nabla_{W_L}f &=\sqrt{m} X_{L-1}\\
\nabla_{X_{L-1}}f &=\sqrt{m} W_{L}\\
\\
\nabla_{W_{L-1}}f &= \nabla_{W_{L-1}}f(X_{L-1}(W_{L-1}, X_{L-2}), W_L)=\nabla_{X_{L-1}}f \cdot \nabla_{W_{L-1}}X_{L-1}(W_{L-1}, X_{L-2})\\
\nabla_{X_{L-2}}f &= \nabla_{X_{L-2}}f(X_{L-1}(W_{L-1}, X_{L-2}), W_L)=\nabla_{X_{L-1}}f \cdot \nabla_{X_{L-2}}X_{L-1}(W_{L-1}, X_{L-2})\\
\\
\nabla_{W_{L-2}}f &= \nabla_{W_{L-2}}f(X_{L-2}(W_{L-2}, X_{L-3}), W_L,W_{L-1})=\nabla_{X_{L-2}}f \cdot \nabla_{W_{L-2}}X_{L-2}(W_{L-2}, X_{L-3})\\
\nabla_{X_{L-3}}f &= \nabla_{X_{L-3}}f(X_{L-2}(W_{L-2}, X_{L-3}), W_L,W_{L-1})=\nabla_{X_{L-2}}f \cdot \nabla_{X_{L-3}}X_{L-2}(W_{L-2}, X_{L-3})\\
\cdots\\
\nabla_{W_{l}}f &= \nabla_{W_{l}}f(X_{l}(W_{l}, X_{l-1}), W_L,W_{L-1},\cdots,W_{l+1})=\nabla_{X_{l}}f \cdot \nabla_{W_{l}}X_{l}(W_{l}, X_{l-1})\\
\nabla_{X_{l-1}}f &= \nabla_{X_{l-1}}f(X_{l}(W_{l}, X_{l-1}), W_L,W_{L-1},\cdots,W_{l+1})=\nabla_{X_{l}}f \cdot \nabla_{X_{l-1}}X_{l}(W_{l}, X_{l-1})\\
\cdots\\
\nabla_{W_{1}}f &= \nabla_{W_{1}}f(X_{1}(W_{1}, X_{0}), W_L,W_{L-1},\cdots,W_{2})=\nabla_{X_{1}}f \cdot \nabla_{W_{1}}X_{1}(W_{1}, X_{0})\\
\nabla_{X_{0}}f &= \nabla_{X_{0}}f(X_{1}(W_{1}, X_{0}), W_L,W_{L-1},\cdots,W_{2})=\nabla_{X_{1}}f \cdot \nabla_{X_{0}}X_{1}(W_{1}, X_{0})\\
\end{align}
$$

$\nabla_{W_{l}}f$ is a matrix, to be specific, $\nabla_{W_{l}}f=\left[\begin{matrix}\frac{\partial f}{\partial w^{(l)}_{11}}&\cdots &\frac{\partial f}{\partial w^{(l)}_{1m}}\\ \vdots&& \vdots\\ \frac{\partial f}{\partial w^{(l)}_{m1}}&\cdots &\frac{\partial f}{\partial w^{(l)}_{mm}}\end{matrix}\right]$

$$
\begin{align}
\left[\begin{matrix}\frac{\partial f}{\partial w^{(l)}_{11}}&\cdots &\frac{\partial f}{\partial w^{(l)}_{1m}}\\ \vdots&& \vdots\\ \frac{\partial f}{\partial w^{(l)}_{m1}}&\cdots &\frac{\partial f}{\partial w^{(l)}_{mm}}\end{matrix}\right]=&

\left[\begin{matrix}\frac{\partial f}{\partial x^{(l)}_{1}} \frac{\partial x^{(l)}_{1}}{\partial w^{(l)}_{11}}&\cdots &\frac{\partial f}{\partial x^{(l)}_{1}}\frac{\partial x^{(l)}_{1}}{\partial w^{(l)}_{1m}}\\ \vdots&& \vdots\\ \frac{\partial f}{\partial x^{(l)}_{m}}\frac{\partial x^{(l)}_{m}}{\partial w^{(l)}_{m1}}&\cdots &\frac{\partial f}{\partial x^{(l)}_{m}}\frac{\partial x^{(l)}_{m}}{\partial w^{(l)}_{mm}}\end{matrix}\right]\\

=&
\left[\begin{matrix}\frac{\partial f}{\partial x^{(l)}_{1}} \mathbb{1}_{\sigma(w^{(l)}_{11}x^{(l-1)}_1+w^{(l)}_{12}x^{(l-1)}_2+\cdots+w^{(l)}_{1m}x^{(l-1)}_m)>0} x^{(l-1)}_{1}&\cdots &\frac{\partial f}{\partial x^{(l)}_{1}}\mathbb{1}_{\sigma(w^{(l)}_{11}x^{(l-1)}_1+w^{(l)}_{12}x^{(l-1)}_2+\cdots+w^{(l)}_{1m}x^{(l-1)}_m)>0} x^{(l-1)}_{m}\\ \vdots&& \vdots\\ \frac{\partial f}{\partial x^{(l)}_{m}}\mathbb{1}_{\sigma(w^{(l)}_{m1}x^{(l-1)}_1+w^{(l)}_{m2}x^{(l-1)}_2+\cdots+w^{(l)}_{mm}x^{(l-1)}_m)>0} x^{(l-1)}_{1}&\cdots &\frac{\partial f}{\partial x^{(l)}_{m}}\mathbb{1}_{\sigma(w^{(l)}_{m1}x^{(l-1)}_1+w^{(l)}_{m2}x^{(l-1)}_2+\cdots+w^{(l)}_{mm}x^{(l-1)}_m)>0} x^{(l-1)}_{m}\end{matrix}\right]\\

=&
\left[\begin{matrix}\frac{\partial f}{\partial x^{(l)}_{1}} \mathbb{1}_{\sigma(w^{(l)}_{11}x^{(l-1)}_1+w^{(l)}_{12}x^{(l-1)}_2+\cdots+w^{(l)}_{1m}x^{(l-1)}_m)>0}\\ \vdots\\ \frac{\partial f}{\partial x^{(l)}_{m}}\mathbb{1}_{\sigma(w^{(l)}_{m1}x^{(l-1)}_1+w^{(l)}_{m2}x^{(l-1)}_2+\cdots+w^{(l)}_{mm}x^{(l-1)}_m)>0}\end{matrix}\right]\left[\begin{matrix}x^{(l-1)}_{1} & x^{(l-1)}_{2}&\cdots&x^{(l-1)}_{m}  \end{matrix}\right]\\
\end{align}
$$

$\nabla_{X_{l-1}}f$ is a vector, to be specific, $\nabla_{X_{l-1}}f=\left[\begin{matrix}\frac{\partial f}{\partial x^{(l-1)}_{1}}\\ \vdots\\ \frac{\partial f}{\partial x^{(l-1)}_{m}}\end{matrix}\right]$
$$
\begin{align}
\left[\begin{matrix}\frac{\partial f}{\partial x^{(l-1)}_{1}}\\ \vdots\\ \frac{\partial f}{\partial x^{(l-1)}_{m}}\end{matrix}\right]=&

\left[\begin{matrix}\frac{\partial f}{\partial x^{(l)}_{1}}\frac{\partial x^{(l)}_1}{\partial x^{(l-1)}_1}+\frac{\partial f}{\partial x^{(l)}_{2}}\frac{\partial x^{(l)}_{2}}{\partial x^{(l-1)}_{1}}+\cdots+\frac{\partial f}{\partial x^{(l)}_{m}}\frac{\partial x^{(l)}_{m}}{\partial x^{(l-1)}_{1}}\\ \vdots\\ \frac{\partial f}{\partial x^{(l)}_{1}}\frac{\partial x^{(l)}_{1}}{\partial x^{(l-1)}_{m}}+\frac{\partial f}{\partial x^{(l)}_{2}}\frac{\partial x^{(l)}_{2}}{\partial x^{(l-1)}_{m}}+\cdots+\frac{\partial f}{\partial x^{(l)}_{m}}\frac{\partial x^{(l)}_{m}}{\partial x^{(l-1)}_{m}}\end{matrix}\right]\\

=&
\left[\begin{matrix}\frac{\partial f}{\partial x^{(l)}_{1}}\mathbb{1}_{\sigma(w^{(l)}_{11}x^{(l-1)}_1+w^{(l)}_{12}x^{(l-1)}_2+\cdots+w^{(l)}_{1m}x^{(l-1)}_m)>0}w^{(l)}_{11}+\frac{\partial f}{\partial x^{(l)}_{2}}\mathbb{1}_{\sigma(w^{(l)}_{21}x^{(l-1)}_1+w^{(l)}_{22}x^{(l-1)}_2+\cdots+w^{(l)}_{2m}x^{(l-1)}_m)>0}w^{(l)}_{21}+\cdots+\frac{\partial f}{\partial x^{(l)}_{m}}\mathbb{1}_{\sigma(w^{(l)}_{m1}x^{(l-1)}_1+w^{(l)}_{m2}x^{(l-1)}_2+\cdots+w^{(l)}_{mm}x^{(l-1)}_m)>0}w^{(l)}_{m1}\\ 
\vdots\\ 
\frac{\partial f}{\partial x^{(l)}_{1}}\mathbb{1}_{\sigma(w^{(l)}_{11}x^{(l-1)}_1+w^{(l)}_{12}x^{(l-1)}_2+\cdots+w^{(l)}_{1m}x^{(l-1)}_m)>0}w^{(l)}_{1m}+\frac{\partial f}{\partial x^{(l)}_{2}}\mathbb{1}_{\sigma(w^{(l)}_{21}x^{(l-1)}_1+w^{(l)}_{22}x^{(l-1)}_2+\cdots+w^{(l)}_{2m}x^{(l-1)}_m)>0}w^{(l)}_{2m}+\cdots+\frac{\partial f}{\partial x^{(l)}_{m}}\mathbb{1}_{\sigma(w^{(l)}_{m1}x^{(l-1)}_1+w^{(l)}_{m2}x^{(l-1)}_2+\cdots+w^{(l)}_{mm}x^{(l-1)}_m)>0}w^{(l)}_{mm}\end{matrix}\right ]\\

=&\left[\begin{matrix}w^{(l)}_{11}&\cdots& w^{(l)}_{1m}\\ \vdots&&\vdots\\ w^{(l)}_{m1}&\cdots& w^{(l)}_{mm}\end{matrix}\right]^T
\left[\begin{matrix}\frac{\partial f}{\partial x^{(l)}_{1}}\mathbb{1}_{\sigma(w^{(l)}_{11}x^{(l-1)}_1+w^{(l)}_{12}x^{(l-1)}_2+\cdots+w^{(l)}_{1m}x^{(l-1)}_m)>0}\\ \vdots\\ \frac{\partial f}{\partial x^{(l)}_{m}}\mathbb{1}_{\sigma(w^{(l)}_{m1}x^{(l-1)}_1+w^{(l)}_{m2}x^{(l-1)}_2+\cdots+w^{(l)}_{mm}x^{(l-1)}_m)>0}\end{matrix}\right]\\
\end{align}
$$

# Gradient of loss function
gradient of loss function:
we set $\mathcal{L}(\theta) = \sum_{i=1}^t\frac{(f(x_{i,a_i};\theta)-r_{i,a_i})^2}{2}+\frac{m\lambda||\theta-\theta^{(0)}||^2_2}{2}=\sum_{i=1}^t \frac{(X^{(i)}_L - r_{i,a_i})^2}{2} + \frac{m\lambda||\theta-\theta^{(0)}||^2_2}{2}$

We can separate the loss function into two parts, the first part is the square term, the second part is the norm term. Define $L_1(\theta)=\sum_{i=1}^t \frac{(X^{(i)}_L - r_{i,a_i})^2}{2}$, $L_2(\theta)=\frac{m\lambda||\theta-\theta^{(0)}||^2_2}{2}$

Then the gradient of square term could be calculated as follows
$$
\begin{align}
\nabla\mathcal{L}(\theta) = \sum_{i=1}^t(f(x_{i,a_i};\theta)-r_{i,a_i})\nabla f(x_{i,a_i};\theta) + m\lambda(\theta-\theta^{(0)})
\end{align}
$$