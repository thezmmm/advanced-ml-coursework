# Exercise 2: Multivariate Calculus in Practice

## Setup

We have N data points

The model prediction is:
$$
\hat{y}_n = f(\mathbf{x}_n^T \boldsymbol{\theta} + b), \qquad f(a) = e^a
$$


The loss is the weighted sum of squared errors:
$$
L = \sum_{n=1}^{N} w_n (y_n - \hat{y}_n)^2, \qquad w_n > 0
$$

------

## (a) Matrix Notation

X - data matrix

θ - parameter vector

b - bias

y - target vector

W - diagonal weight matrix



Introduce the following matrices and vectors:### Vector-valued

Extend f to act element-wise on vectors. For a in R^N
$$
f(\mathbf{a}) = \begin{pmatrix} e^{a_1} \\ \vdots \\ e^{a_N} \end{pmatrix} \in \mathbb{R}^N
$$
The pre-activation vector and prediction vector are:
$$
\mathbf{a} = \mathbf{X}\boldsymbol{\theta} + b\mathbf{1} \in \mathbb{R}^N, \qquad \hat{\mathbf{y}} = f(\mathbf{a}) \in \mathbb{R}^N
$$


### Loss in matrix form

$$
\boxed{L = (\mathbf{y} - \hat{\mathbf{y}})^T \mathbf{W} (\mathbf{y} - \hat{\mathbf{y}})}
$$



**Verification of dimensions:**
$$
\underbrace{(\mathbf{y} - \hat{\mathbf{y}})^T}*{1 \times N} \underbrace{\mathbf{W}}*{N \times N} \underbrace{(\mathbf{y} - \hat{\mathbf{y}})}_{N \times 1} = \text{scalar} 
$$


This equals
$$
\sum_{n=1}^N w_n(y_n - \hat{y}_n)^2
$$


------

## (b) Jacobian and Gradients

### Jacobian of vector-valued $f$

The Jacobian of f  is the N\*N matrix of partial derivatives:


$$
J_f(\mathbf{a}) = \frac{\partial f(\mathbf{a})}{\partial \mathbf{a}} = \begin{pmatrix} \frac{\partial f_1}{\partial a_1} & \cdots & \frac{\partial f_1}{\partial a_N} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_N}{\partial a_1} & \cdots & \frac{\partial f_N}{\partial a_N} \end{pmatrix}
$$


Since
$$
f_i(\mathbf{a}) = e^{a_i}
$$
 only depends on ai, the Jacobian is diagonal:
$$
\boxed{J_f(\mathbf{a}) = \text{diag}(e^{a_1}, \ldots, e^{a_N}) = \text{diag}(\hat{\mathbf{y}}) \in \mathbb{R}^{N \times N}}
$$


### Chain rule setup

Define the residual vector
$$
\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}} \in \mathbb{R}^N
$$
so
$$
L = \mathbf{r}^T \mathbf{W} \mathbf{r}
$$

### Gradient of L

$$
\frac{\partial L}{\partial \hat{\mathbf{y}}} = -2\mathbf{W}\mathbf{r} = -2\mathbf{W}(\mathbf{y} - \hat{\mathbf{y}})
$$


$$
\frac{\partial L}{\partial \mathbf{a}} = J_f(\mathbf{a})^T \frac{\partial L}{\partial \hat{\mathbf{y}}} = \text{diag}(\hat{\mathbf{y}}) \cdot \bigl(-2\mathbf{W}(\mathbf{y} - \hat{\mathbf{y}})\bigr)
$$


Since
$$
\mathbf{a} = \mathbf{X}\boldsymbol{\theta} + b\mathbf{1}
$$

$$
\frac{\partial \mathbf{a}}{\partial \boldsymbol{\theta}} = \mathbf{X}
$$
so:
$$
\boxed{\nabla_{\boldsymbol{\theta}} L = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{a}} = -2,\mathbf{X}^T ,\text{diag}(\hat{\mathbf{y}}), \mathbf{W},(\mathbf{y} - \hat{\mathbf{y}})}
$$


Since
$$
\frac{\partial \mathbf{a}}{\partial b} = \mathbf{1}
$$

$$
\boxed{\nabla_b L = \mathbf{1}^T \frac{\partial L}{\partial \mathbf{a}} = -2,\mathbf{1}^T ,\text{diag}(\hat{\mathbf{y}}), \mathbf{W},(\mathbf{y} - \hat{\mathbf{y}})}
$$
