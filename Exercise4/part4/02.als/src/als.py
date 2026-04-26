import sys
import numpy as np


def error(X, W, H, reg):
    mask = ~np.isnan(X)
    diff = (X - W @ H)[mask]
    err = np.sum(diff**2) + reg * (np.sum(W**2) + np.sum(H**2))
    return err


def solve(X, W, reg):
    m = X.shape[1]
    k = W.shape[1]
    H = np.zeros((k, m))

    for i in range(m):
        xi = X[:, i]
        present = ~np.isnan(xi)
        xi_obs = xi[present]
        Wi = W[present, :]
        H[:, i] = np.linalg.solve(Wi.T @ Wi + reg * np.eye(k), Wi.T @ xi_obs)

    return H


def als(X, W, reg, itercnt):
    err = np.zeros(itercnt)

    for i in range(itercnt):
        H = solve(X, W, reg)
        W = solve(X.T, H.T, reg).T
        err[i] = error(X, W, H, reg)

    return W, H, err


def main(argv):
    np.random.seed(2022)

    X = np.genfromtxt(argv[1])
    k = int(argv[2])
    reg = float(argv[3])
    itercnt = int(argv[4])

    n, m = X.shape
    W = np.random.random((n, k))

    W, H, err = als(X, W, reg, itercnt)
    print('W')
    print(W)
    print('H')
    print(H)
    print('errors')
    print(err)
    print('product matrix')
    print(W @ H)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('usage: python %s filename number_of_factors regularizer iteration_count' % sys.argv[0])
    else:
        main(sys.argv)