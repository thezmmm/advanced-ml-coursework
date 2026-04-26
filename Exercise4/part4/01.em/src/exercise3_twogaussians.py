import numpy as np
import matplotlib.pyplot as plt
from part4.em import computeresponsibilities, computeparameters, computeparameterssame, em
# Load data
data = np.loadtxt('2gaussians.txt')  # Replace with actual path if needed
X = data

# Initialize responsibilities randomly
np.random.seed(42)
N = X.shape[0]
K = 2
R_init = np.random.dirichlet(np.ones(K), size=N)

# --- Standard mixture model ---
R1 = R_init.copy()
_, _, _, R1_final = em(X, R1, 100, computeparameters)

# --- Shared covariance model ---
R2 = R_init.copy()
_, _, _, R2_final = em(X, R2, 100, computeparameterssame)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=R1_final[:, 0], cmap='coolwarm', alpha=0.7)
axes[0].set_title('Standard GMM (full covariance)')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

axes[1].scatter(X[:, 0], X[:, 1], c=R2_final[:, 0], cmap='coolwarm', alpha=0.7)
axes[1].set_title('Shared covariance GMM')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')

plt.tight_layout()
plt.savefig('exercise3_plots.pdf')
plt.show()
print("Saved exercise3_plots.pdf")