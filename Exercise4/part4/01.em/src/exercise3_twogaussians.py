import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')

from em import computeresponsibilities, computeparameters, computeparameterssame, em

# Load data
X = np.loadtxt('2gaussians.txt')

# Initialize responsibilities randomly
np.random.seed(42)
N = X.shape[0]
K = 2
R_init = np.random.rand(N, K)
R_init = R_init / R_init.sum(axis=1)[:, np.newaxis]

# Standard mixture model
R1_final, prior1, mu1, C1 = em(X, R_init.copy(), 100, computeparameters)

# Shared covariance model
R2_final, prior2, mu2, C2 = em(X, R_init.copy(), 100, computeparameterssame)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc1 = axes[0].scatter(X[:, 0], X[:, 1], c=R1_final[:, 0], cmap='coolwarm', alpha=0.7)
axes[0].set_title('Standard GMM (full covariance)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
plt.colorbar(sc1, ax=axes[0], label='Responsibility (cluster 1)')

sc2 = axes[1].scatter(X[:, 0], X[:, 1], c=R2_final[:, 0], cmap='coolwarm', alpha=0.7)
axes[1].set_title('Shared covariance GMM')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
plt.colorbar(sc2, ax=axes[1], label='Responsibility (cluster 1)')

plt.tight_layout()
plt.savefig('answer3.png')
plt.show()
print("Saved answer3.png")