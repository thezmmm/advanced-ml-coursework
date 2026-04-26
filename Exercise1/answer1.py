import numpy as np
import matplotlib.pyplot as plt

def true_risk(alpha):
    """
    R(alpha) = (28/3)*alpha^2 - 56*alpha + 253/3
      x ~ Uniform(-2, 6)  =>  E[x^2] = (a^2+ab+b^2)/3 = 28/3
      y|x ~ Uniform(3x-1, 3x+1)  =>  E[y|x]=3x, Var(y|x)=1/3
      R(alpha) = E[(y - alpha*x)^2]
               = Var(y|x) + E[(3x - alpha*x)^2]
               = 1/3 + (3 - alpha)^2 * E[x^2]
               = 1/3 + (3 - alpha)^2 * (28/3)
         (28/3)*alpha^2 - 56*alpha + 253/3
    best alpha* = 3, R(3) = 1/3
    """
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

M_values = np.logspace(1, 5, 80).astype(int)  # M 从 10 到 100000

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