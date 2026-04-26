import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances


def sammon(D, P, eta, tau, itercnt):
    cnt = D.shape[0]
    eps = 1e-10
    total = np.sum(D) / 2.0  # sum_{i<j} d'_ij

    for r in range(itercnt):
        d = euclidean_distances(P)

        with np.errstate(divide='ignore', invalid='ignore'):
            W = (d - D) / (D * d + eps)
        np.fill_diagonal(W, 0.0)

        row_sum = W.sum(axis=1)
        grad = 2.0 * (row_sum[:, None] * P - W @ P) / total

        step = eta / (1.0 + tau * eta * r)
        P = P - step * grad

    return P

def main(argv):
    X = np.loadtxt(argv[1])
    k = int(argv[2])
    eta = float(argv[3])
    tau = float(argv[4])
    itercnt = int(argv[5])

    D = euclidean_distances(X)

    pca = PCA(n_components=k)
    P = pca.fit_transform(X)
    print('PCA:')
    print(P)

    print('Sammon:')
    print(sammon(D, P, eta, tau, itercnt))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('usage: python %s filename proj_dim eta tau itercnt' % sys.argv[0])
    else:
        main(sys.argv)