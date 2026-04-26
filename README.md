# Advanced Course in Machine Learning — Coursework

University of Helsinki · Spring 2026

From-scratch implementations of core machine learning algorithms in pure NumPy, graded via the [TMC](https://tmc.mooc.fi/) platform. Each programming exercise follows the pattern `src/<algorithm>.py` (implementation) + `test/` (TMC tests).

---

## Exercises

### Exercise 1 — Optimization & Dimensionality Reduction

| Topic | File | Description |
|---|---|---|
| Gradient Descent | `Exercise1/gradient/src/gradient.py` | Mini-batch SGD with deterministic decay and AdaGrad step sizes |
| PCA | `Exercise1/pca/src/pca.py` | Principal Component Analysis via covariance matrix + eigendecomposition (top-2 components) |

**Written answers:** Q1 (`answer1.md`), Q2 (`answer2.md`), Q5 (`answer5.md`)  
**Data:** Finnish 2022 parliamentary election votes (`elec2022.txt`, `parties.txt`)

---

### Exercise 2 — Probabilistic & Regularized Linear Models

| Topic | File | Description |
|---|---|---|
| Naive Bayes | `Exercise2/part2/01.nb/src/nb.py` | Bernoulli Naive Bayes; derives equivalent linear weights `w` and bias `b` |
| Logistic Regression | `Exercise2/part2/02.irls/src/irls.py` | IRLS (Newton's method); tracks negative log-likelihood and misclassification rate |
| Lasso | `Exercise2/part2/03.lasso/src/lasso.py` | Coordinate descent with soft-thresholding |

**Written answers:** Q1 (`answer1.md`), Q2 (`answer2.md`)

---

### Exercise 3 — Discriminative & Ensemble Models

| Topic | File | Description |
|---|---|---|
| Linear SVM | `Exercise3/part3/01.svm/src/svm_linear.py` | SMO algorithm on dual variables `α`; linear kernel |
| Kernel SVM | `Exercise3/part3/01.svm/src/svm_kernel.py` | Same SMO with pluggable kernel (linear / RBF) |
| Random Forest | `Exercise3/part3/02.rf/src/rf.py` | Bootstrap sampling + random feature subsets; entropy-gain decision trees |
| AdaBoost | `Exercise3/part3/03.adaboost/src/adaboost.py` | LDA weak learners; tracks individual, ensemble, and exponential loss per round |
| Neural Network | `Exercise3/part3/04.neural/src/nn.py` | Graph-based feed-forward net (`Edge`/`Neuron`/`NN`); sigmoid activations; backpropagation |

**Written answer:** Q4 (`answer4.pdf`)

---

### Exercise 4 — Latent Variable Models & Manifold Learning

| Topic | File | Description |
|---|---|---|
| EM / GMM | `Exercise4/part4/01.em/src/em.py` | EM for Gaussian Mixture Models; four covariance modes: full, diagonal, shared, spherical |
| ALS | `Exercise4/part4/02.als/src/als.py` | Alternating Least Squares for matrix factorization with missing values and L2 regularization |
| NMF | `Exercise4/part4/03.nmf/src/nmf.py` | Non-negative Matrix Factorization via multiplicative update rules |
| Sammon Mapping | `Exercise4/part4/04.sammon/src/sammon.py` | Gradient descent on Sammon stress; PCA-initialized projection |

**Written answers:** Q1 (`answer1.md`), Q3 (`answer3.md`, includes GMM visualization)  
**Data:** Two-Gaussians dataset (`2gaussians.txt`)

---

## Running Tests

Each exercise uses pytest via the TMC runner. From any `01.xxx/` directory:

```bash
pytest test/
```

Or run the standalone script directly:

```bash
# Example: Gradient descent
python Exercise1/gradient/src/gradient.py Exercise1/gradient/src/toy.txt determ 0.1 0.01 10 5

# Example: PCA
python Exercise1/pca/src/pca.py Exercise1/pca/src/toy.txt

# Example: Naive Bayes
python Exercise2/part2/01.nb/src/nb.py Exercise2/part2/01.nb/src/toy.txt

# Example: Kernel SVM (RBF)
python Exercise3/part3/01.svm/src/svm_kernel.py Exercise3/part3/01.svm/src/toy.txt 1.0

# Example: EM (full covariance, 2 clusters, 20 iterations)
python Exercise4/part4/01.em/src/em.py Exercise4/part4/01.em/src/toy.txt 2 normal 20
```

---

## Dependencies

| Package | Used in |
|---|---|
| `numpy` | All exercises |
| `scipy` | Exercise 4 — EM (`multivariate_normal.logpdf`) |
| `sklearn` | Exercise 4 — Sammon (`euclidean_distances`, PCA init) |
| `matplotlib` | Exercise 4 — GMM visualization (`answer3.png`) |

Install:

```bash
pip install numpy scipy scikit-learn matplotlib
```