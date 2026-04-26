import sys
import numpy as np
from scipy.stats import multivariate_normal


def logsumrows(X):
    """
    Computes the sums of rows of log-numbers
    """
    M = -np.max(X, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(X + M), axis=1)) - M.squeeze()


def computeparameters(R, X):
    """
    Computes the optimal parameters for the Gaussian mixture model
    """
    k = R.shape[1]
    dim = X.shape[1]
    n = X.shape[0]

    prior = np.zeros(k)
    mu = np.zeros((k, dim))
    C = np.zeros((k, dim, dim))

    Nk = R.sum(axis=0)          # shape (k,)
    prior = Nk / n
    mu = (R.T @ X) / Nk[:, np.newaxis]   # shape (k, dim)

    for j in range(k):
        diff = X - mu[j]        # shape (n, dim)
        C[j] = (R[:, j:j+1] * diff).T @ diff / Nk[j]

    return prior, mu, C


def computeparametersdiagonal(R, X):
    """
    Computes the optimal parameters for the Gaussian mixture model with
    diagonal covariance matrices.
    """
    k = R.shape[1]
    dim = X.shape[1]
    n = X.shape[0]

    prior = np.zeros(k)
    mu = np.zeros((k, dim))
    C = np.zeros((k, dim, dim))

    Nk = R.sum(axis=0)
    prior = Nk / n
    mu = (R.T @ X) / Nk[:, np.newaxis]

    for j in range(k):
        diff = X - mu[j]                              # (n, dim)
        var = np.einsum('n,nd->d', R[:, j], diff**2) / Nk[j]  # (dim,)
        C[j] = np.diag(var)

    return prior, mu, C


def computeparameterssame(R, X):
    """
    Computes the optimal parameters for the Gaussian mixture model with
    equal covariance matrices.
    """
    k = R.shape[1]
    cnt, dim = X.shape

    prior = np.zeros(k)
    mu = np.zeros((k, dim))
    C = np.zeros((k, dim, dim))

    Nk = R.sum(axis=0)
    prior = Nk / cnt
    mu = (R.T @ X) / Nk[:, np.newaxis]

    Sigma = np.zeros((dim, dim))
    for j in range(k):
        diff = X - mu[j]
        Sigma += (R[:, j:j+1] * diff).T @ diff

    Sigma /= cnt         # normalize by N, not N_k
    C[:] = Sigma         # all clusters share the same matrix

    return prior, mu, C


def computeparametersspherical(R, X):
    """
    Computes the optimal parameters for the Gaussian mixture model with
    equal diagonal spherical covariance matrices.
    """
    k = R.shape[1]
    cnt, dim = X.shape

    prior = np.zeros(k)
    mu = np.zeros((k, dim))
    C = np.zeros((k, dim, dim))

    Nk = R.sum(axis=0)
    prior = Nk / cnt
    mu = (R.T @ X) / Nk[:, np.newaxis]

    sigma2 = 0.0
    for j in range(k):
        diff = X - mu[j]
        sigma2 += np.sum(R[:, j:j+1] * diff**2)

    sigma2 /= (cnt * dim)   # normalize by N*D
    C[:] = sigma2 * np.eye(dim)

    return prior, mu, C


def computeresponsibilities(X, prior, mu, C):
    """
    Computes responsibilities using log-space to avoid overflow.
    """
    k = prior.shape[0]
    cnt = X.shape[0]

    L = np.zeros((cnt, k))
    for j in range(k):
        L[:, j] = np.log(prior[j]) + multivariate_normal.logpdf(X, mean=mu[j], cov=C[j])

    log_px = logsumrows(L)                      # shape (cnt,)
    R = np.exp(L - log_px[:, np.newaxis])
    return R


def em(X, R, itercnt, stats):
    """
    EM algorithm main loop.
    """
    prior, mu, C = None, None, None

    for _ in range(itercnt):
        prior, mu, C = stats(R, X)       # M-step
        R = computeresponsibilities(X, prior, mu, C)   # E-step

    return R, prior, mu, C


def main(argv):
    np.random.seed(2022)

    X = np.loadtxt(argv[1])
    k = int(argv[2])
    mode = argv[3]
    itercnt = int(argv[4])

    n, m = X.shape
    R = np.random.rand(X.shape[0], k)
    R = R / R.sum(axis=1)[:, np.newaxis]
    print(R)

    if mode == 'normal':
        R, prior, mu, C = em(X, R, itercnt, computeparameters)
    elif mode == 'diag':
        R, prior, mu, C = em(X, R, itercnt, computeparametersdiagonal)
    elif mode == 'same':
        R, prior, mu, C = em(X, R, itercnt, computeparameterssame)
    elif mode == 'sphere':
        R, prior, mu, C = em(X, R, itercnt, computeparametersspherical)
    else:
        print("Mode %s unrecognized" % mode)
        return

    print('R')
    print(R)
    print('priors')
    print(prior)
    print('means')
    print(mu)
    print('covariance matrices')
    print(C)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('usage: python %s filename number_of_factors iteration_count' % sys.argv[0])
    else:
        main(sys.argv)