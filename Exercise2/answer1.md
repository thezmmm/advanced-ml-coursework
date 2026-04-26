## Logistic regression

### a

Logistic regression model definition
$$
p(y=1 \mid w, x)=\sigma\left(w^{T} x\right)=\frac{1}{1+e^{-w^{T} x}}
$$
So
$$
p(y \mid w, x)=\sigma\left(yw^{T} x\right)=\frac{1}{1+e^{-yw^{T} x}}
$$

$$
-\sum_{n} \log p\left(y_{n} \mid w, x_{n}\right)=-\sum_{n} \log \frac{1}{1+e^{-y_{n} w^{T} x_{n}}}=\sum_{n} \log \left(1+e^{-y_{n} w^{T} x_{n}}\right)
$$

### b

Define Loss
$$
L(w) =\sum_{n} \log \left(1+e^{-y_{n} w^{T} x_{n}}\right)
$$
**get gradient G**

when yn = 1
$$
\frac{\partial}{\partial w} \log \left(1+e^{-w^{T} x_{n}}\right)=\frac{-x_{n} e^{-w^{T} x_{n}}}{1+e^{-w^{T} x_{n}}}=-x_{n}\left(1-\mu_{n}\right)=x_{n}\left(\mu_{n}-1\right)
$$
when yn = -1
$$
\frac{\partial}{\partial w} \log \left(1+e^{w^{T} x_{n}}\right)=\frac{-x_{n} e^{w^{T} x_{n}}}{1+e^{w^{T} x_{n}}}=x_{n}\mu_{n}
$$
so, for any yn
$$
\frac{\partial}{\partial w} \log \left(1+e^{-y_nw^{T} x_{n}}\right) =x_{n}\left(\mu_{n}-\frac{1+y_n}{2}\right)
$$

$$
g=\sum_{n} x_{n}\left(\mu_{n}-\frac{y_{n}+1}{2}\right)=X^{T}(\mu-(y+1) / 2)
$$

**get Hessian H**

Differentiating the Nth item with respect to w
$$
\frac{\partial (\mu_{n}-c_n)x_n}{\partial w}=\frac{\partial \sigma\left(w^{T} x_{n}\right)}{\partial w}=\sigma\left(w^{T} x_{n}\right)\left(1-\sigma\left(w^{T} x_{n}\right)\right) \cdot x_{n}=\mu_{n}\left(1-\mu_{n}\right) x_{n}
$$
Therefore, each contribution to Hessian is
$$
\mu_{n}\left(1-\mu_{n}\right) x_{n}x^T_n=S_{nn}x_{n}x^T
$$
sum
$$
H=\sum_n\mu_{n}\left(1-\mu_{n}\right) x_{n}x^T_n=X^TSX
$$

### c

for any vector v
$$
v^{T} H v=v^{T} X^{T} S X v=(X v)^{T} S(X v)
$$
make u = Xv
$$
v^{T} H v=u^Tsu=\sum_nS_{nn}u^2_n=\sum_n\mu_{n}\left(1-\mu_{n}\right) u_n^2
$$
cause
$$
0<\mu_n<1,u_n^2>0
$$

$$
v^{T} H v=\sum_n\mu_{n}\left(1-\mu_{n}\right) u_n^2 >= 0
$$

So
$$
H >= 0
$$
H is positive semidefinite

Hessian semidefiniteness implies that the loss function is convex. This guarantees that:

- Any local minimum is also a global minimum;

- Optimization algorithms such as gradient descent converge to the global optimum;

- The update direction of Newton's method (IRLS) is always the descent direction.