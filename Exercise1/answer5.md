## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load data
data = np.loadtxt('elec2022.txt')
party_ids   = data[:, 0].astype(int)
answers     = data[:, 1:]

party_names = {}
with open('parties.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        pid  = int(parts[0])
        name = ' '.join(parts[1:])
        party_names[pid] = name

pca = PCA(n_components=2)
Z = pca.fit_transform(answers)

unique_ids = np.unique(party_ids)
party_means = {}
for pid in unique_ids:
    mask = (party_ids == pid)
    party_means[pid] = Z[mask].mean(axis=0)   # shape (2,)

# plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(Z[:, 0], Z[:, 1], c='lightgray', s=15, alpha=0.5, zorder=1)

colors = plt.cm.tab20.colors
for i, pid in enumerate(unique_ids):
    mx, my = party_means[pid]
    ax.scatter(mx, my, s=120, color=colors[i % len(colors)], zorder=3,
               edgecolors='black', linewidths=0.5)
    name = party_names.get(pid, str(pid))
    ax.annotate(name, (mx, my),
                textcoords='offset points', xytext=(6, 4),
                fontsize=8, color=colors[i % len(colors)], fontweight='bold')

ax.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('2022 Western Uusimaa County Elections — PCA of candidate answers\n(party averages annotated)')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.axvline(0, color='gray', lw=0.5, ls='--')
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('answer5_pca_map.png', dpi=150, bbox_inches='tight')
plt.show()
```

## img

![answer5_pca_map](E:\Content\University of Helsinki\Advanced Course in Machine Learning\Exercise1\answer5_pca_map.png)