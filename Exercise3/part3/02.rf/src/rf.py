import sys
import numpy as np


def entropy(C):
    with np.errstate(divide='ignore'):
        L = np.log(C)
        N = np.log(np.sum(C, axis=0, keepdims=True))
    L[np.isinf(L)] = 0
    N[np.isinf(N)] = 0
    return -np.sum(C * (L - N), axis=0)


def entropy_gain(X, y):
    cnt, k = X.shape
    labels = (y + 1) / 2
    m = np.sum(labels)

    P1 = np.zeros((2, k))
    P0 = np.zeros((2, k))

    cnt1 = np.sum(X, axis=0)
    cnt0 = cnt - cnt1

    P1[1, :] = labels @ X
    P1[0, :] = cnt1 - P1[1, :]
    P0[1, :] = m - P1[1, :]
    P0[0, :] = cnt0 - P0[1, :]

    Q = np.array([cnt - m, m])

    return entropy(Q) - entropy(P1) - entropy(P0)


def fit(X, y, featurecnt):
    g = entropy_gain(X, y)

    feasible = np.where(g > 0.00001)[0]

    if len(feasible) == 0:
        return int(np.sign(0.5 + np.sum(y)))

    sample_size = min(featurecnt, len(feasible))
    sampled = np.random.choice(feasible, size=sample_size, replace=False)

    ind = sampled[np.argmax(g[sampled])]

    split = X[:, ind] == 1
    return (ind,
            fit(X[~split, :], y[~split], featurecnt),
            fit(X[split, :], y[split], featurecnt))


def predict_sample(T, x):
    if type(T) is tuple:
        ind = int(x[T[0]])
        return predict_sample(T[1 + ind], x)
    return T


def predict(T, X):
    cnt = X.shape[0]
    p = np.zeros(cnt)
    for i in range(cnt):
        p[i] = predict_sample(T, X[i, :])
    return p


def rf(Xtrain, ltrain, treecnt, samplecnt, featurecnt, Xtest, ltest):
    cnt, k = Xtrain.shape

    p = np.zeros(Xtest.shape[0])
    misclass = np.zeros(treecnt)

    for i in range(treecnt):
        idx = np.random.choice(cnt, size=samplecnt, replace=True)
        X_boot = Xtrain[idx, :]
        y_boot = ltrain[idx]

        tree = fit(X_boot, y_boot, featurecnt)

        p += predict(tree, Xtest)

        labels = np.sign(p)
        misclass[i] = np.mean(labels != ltest)

    return p, misclass


def main(argv):
    np.random.seed(2022)

    D = np.loadtxt(argv[1])
    ltrain = D[:, 0]
    Xtrain = D[:, 1:]

    D = np.loadtxt(argv[2])
    ltest = D[:, 0]
    Xtest = D[:, 1:]

    treecnt = int(argv[3])

    cnt, k = Xtrain.shape

    p, misclass = rf(Xtrain, ltrain, treecnt, cnt, int(np.sqrt(k)), Xtest, ltest)

    print('Votes for testing data: negative -> -1 class, positive -> 1 class')
    print(p)
    print('Misclassification rate')
    print(misclass)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('usage: python %s train_filename test_filename number_of_trees' % sys.argv[0])
    else:
        main(sys.argv)