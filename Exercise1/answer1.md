# Exercise 1: Risk

## Setup

Model: 
$$
\hat{y} = \alpha x,  L(y, x, \alpha) = (y - \alpha x)^2
$$
Data-generating distribution:

- x (-2,, 6)
- y (3x-1,, 3x+1)

## (a) Definition and Computation of Risk

The **risk** is the expected loss under the data-generating distribution:
$$
R(\alpha) = \mathbb{E}_{p(y,x)}\bigl[(y - \alpha x)^2\bigr]
$$


Since :
$$
y \mid x \sim \text{Uniform}(3x-1,, 3x+1)
$$
So
$$
\mathbb{E}[y \mid x] = \frac{(3x-1)+(3x+1)}{2} = 3x
$$

$$
\text{Var}(y \mid x) = \frac{(3x+1 - (3x-1))^2}{12} = \frac{4}{12} = \frac{1}{3}
$$


$$
\mathbb{E}[y^2 \mid x] = \text{Var}(y \mid x) + \bigl(\mathbb{E}[y \mid x]\bigr)^2 = \frac{1}{3} + 9x^2
$$

$$
\mathbb{E}[x^k] = \frac{1}{b-a}\int_a^b x^k,dx
$$

$$
\mathbb{E}[x] = \frac{-2+6}{2} = 2
$$

$$
\mathbb{E}[x^2] = \frac{(-2)^2 + (-2)(6) + 6^2}{3} = \frac{4 - 12 + 36}{3} = \frac{28}{3}
$$

$$
R(\alpha) = \mathbb{E}\bigl[y^2 - 2\alpha xy + \alpha^2 x^2\bigr] = \mathbb{E}[y^2] - 2\alpha,\mathbb{E}[xy] + \alpha^2,\mathbb{E}[x^2]
$$

Compute each term using the law of total expectation:
$$
\mathbb{E}[y^2] = \mathbb{E}_x\bigl[\mathbb{E}[y^2 \mid x]\bigr] = \mathbb{E}_x!\left[\frac{1}{3} + 9x^2\right]  = \frac{253}{3}
$$

$$
\mathbb{E}[xy] = \mathbb{E}_x\bigl[x,\mathbb{E}[y \mid x]\bigr] = \mathbb{E}_x[3x^2] = 3\cdot\frac{28}{3} = 28
$$

$$
\mathbb{E}[x^2] = \frac{28}{3}
$$

Substituting:
$$
R(\alpha) = \frac{253}{3} - 2\alpha \cdot 28 + \alpha^2 \cdot \frac{28}{3} = \frac{28}{3}\alpha^2 - 56\alpha + \frac{253}{3}
$$
R(α) is a convex quadratic in α

 Setting the derivative to zero
$$
\frac{dR}{d\alpha} = \frac{56}{3}\alpha - 56 = 0 \quad\Longrightarrow\quad \boxed{\alpha^* = 3}
$$
The minimum risk is:
$$
R(3) = \boxed{\frac{1}{3}}
$$

## (b)

```python
import numpy as np
import matplotlib.pyplot as plt

def true_risk(alpha):
    return (28/3)*alpha**2 - 56*alpha + 253/3

def sample_data(M, rng=None):
    """get M points from p(y,x) """
    if rng is None:
        rng = np.random.default_rng()
    x = rng.uniform(-2, 6, M)          # x ~ Uniform(-2, 6)
    y = rng.uniform(3*x - 1, 3*x + 1)  # y|x ~ Uniform(3x-1, 3x+1)
    return x, y


def mc_risk(alpha, M, rng=None):
    """
      R_hat = (1/M) * sum_m (y_m - alpha*x_m)^2
    """
    x, y = sample_data(M, rng)
    losses = (y - alpha * x) ** 2
    return np.mean(losses)


alpha_eval = 1.5
true_val   = true_risk(alpha_eval)
print(f"True R({alpha_eval}) = {true_val:.6f}")

M_values = np.logspace(1, 5, 80).astype(int) 

rng = np.random.default_rng(seed=42)
mc_estimates = [mc_risk(alpha_eval, M, rng) for M in M_values]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(M_values, mc_estimates, color="#4C72B0", lw=1.5, label=r"$\hat{R}(1.5)$ Monte Carlo")
ax.axhline(true_val, color="#DD8452", lw=1.8, ls="--", label=f"True $R(1.5) = {true_val:.4f}$")
ax.set_xscale("log")
ax.set_xlabel("Number of samples $M$")
ax.set_ylabel("Risk estimate")
ax.set_title(r"Monte Carlo estimate of $R(1.5)$ as a function of $M$")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("Figure saved.")
```

## (c)

![img1](E:\Content\University of Helsinki\Advanced Course in Machine Learning\Exercise1\img1.png)