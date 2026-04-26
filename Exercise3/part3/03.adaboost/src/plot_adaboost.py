import numpy as np
import matplotlib.pyplot as plt
from adaboost import adaboost

# ── Load data ─────────────────────────────────────────────────────────────────
D = np.loadtxt('toy.txt')
labels = D[:, 0].copy()
D[:, 0] = 1         # replace label column with bias term (same as main())
X = D
y = labels
X_xy = D[:, 1:]     # original 2-D coords for scatter plots

# ── Run AdaBoost for 100 iterations ──────────────────────────────────────────
output, err_ind, err_ens, err_exp = adaboost(X, y, 100)
iters = np.arange(1, 101)

# ══════════════════════════════════════════════════════════════════════════════
# Figures 1-3: Error curves
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(iters, err_ind, color='steelblue', linewidth=1.5)
axes[0].set_title('Individual classifier weighted error')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Weighted error')
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, err_ens, color='tomato', linewidth=1.5)
axes[1].set_title('Ensemble misclassification error')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Misclassification rate')
axes[1].grid(True, alpha=0.3)

axes[2].plot(iters, err_exp, color='seagreen', linewidth=1.5)
axes[2].set_title('Normalised exponential loss')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Exponential loss')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Figures 4-7: Scatter plots at 8, 20, 50, 100 iterations
# ══════════════════════════════════════════════════════════════════════════════
scatter_iters = [8, 20, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
axes = axes.flatten()

for ax, it in zip(axes, scatter_iters):
    out_t, _, _, _ = adaboost(X, y, it)
    pred = np.sign(out_t)

    correct   = pred == y
    incorrect = ~correct

    # Correctly classified points: circles (+1) and squares (-1)
    for label_val, marker in [(1, 'o'), (-1, 's')]:
        mask = correct & (y == label_val)
        color = 'steelblue' if label_val == 1 else 'tomato'
        ax.scatter(X_xy[mask, 0], X_xy[mask, 1],
                   marker=marker, c=color, edgecolors='k',
                   linewidths=0.4, s=40, alpha=0.85,
                   label=f'Correct (true={label_val:+d})')

    # Misclassified points: cross markers
    for label_val in [1, -1]:
        mask = incorrect & (y == label_val)
        color = 'steelblue' if label_val == 1 else 'tomato'
        ax.scatter(X_xy[mask, 0], X_xy[mask, 1],
                   marker='x', c=color, s=60, linewidths=1.5,
                   label=f'Wrong (true={label_val:+d})')

    err_rate = np.mean(pred != y)
    ax.set_title(f'Iteration {it}  (error={err_rate:.3f})')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()