import sys
import numpy as np


def nmf(X, W, H, itercnt):
    eps = 1e-10
    err = np.zeros(itercnt + 1)
    err[0] = np.sum((X - W @ H)**2)

    for i in range(itercnt):
        # Update H
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        # Update W
        W *= (X @ H.T) / (W @ H @ H.T + eps)

        err[i + 1] = np.sum((X - W @ H)**2)

    return W, H, err


def main(argv):
    np.random.seed(2022)

    X = np.loadtxt(argv[1])
    k = int(argv[2])
    itercnt = int(argv[3])

    n, m = X.shape

    W = np.random.random((n, k))
    H = np.random.random((k, m))

    W, H, err = nmf(X, W, H, itercnt)
    print('W')
    print(W)
    print('H')
    print(H)
    print('errors')
    print(err)
    print('product matrix')
    print(W @ H)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('usage: python %s filename number_of_factors iteration_count' % sys.argv[0])
    else:
        main(sys.argv)