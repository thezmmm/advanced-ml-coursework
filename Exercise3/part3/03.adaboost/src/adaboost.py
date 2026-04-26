import sys
import numpy as np

class LDA:
    def fit(self, X, y, w):
        C = np.linalg.inv(np.cov(X.T, aweights=w) + 0.001*np.eye(X.shape[1]))
        ind0 = y == -1
        ind1 = y == 1
        m0 = np.sum(X[ind0, :] * w[ind0, np.newaxis], axis=0) / np.sum(w[ind0])
        m1 = np.sum(X[ind1, :] * w[ind1, np.newaxis], axis=0) / np.sum(w[ind1])

        self.w = C @ (m1 - m0)
        self.b = self.find_threshold(X, y, w)
        if self.b == None:
            self.w = -self.w
            self.b = self.find_threshold(X, y, w)

    def find_threshold(self, X, y, w):
        score = X @ self.w
        ind = np.argsort(score)
        err = np.cumsum(y[ind]*w[ind])
        i = np.argmin(err)
        if i == len(err) - 1:
            return None
        return (score[ind[i]] + score[ind[i + 1]]) / 2

    def predict(self, X):
        return np.sign(X @ self.w - self.b)


def adaboost(X, y, itercnt):
    cnt, k = X.shape

    err_individual = np.zeros(itercnt)
    err_ensemble = np.zeros(itercnt)
    err_exponential = np.zeros(itercnt)
    output = np.zeros(cnt)

    w = np.ones(cnt) / cnt

    for t in range(itercnt):
        clf = LDA()
        clf.fit(X, y, w)
        pred = clf.predict(X)

        incorrect = (pred != y).astype(float)
        eps = np.dot(w, incorrect)
        err_individual[t] = eps

        eps = np.clip(eps, 1e-10, 1 - 1e-10)

        alpha = 0.5 * np.log((1 - eps) / eps)

        output += alpha * pred

        w = w * np.exp(-alpha * y * pred)
        w /= w.sum()

        err_ensemble[t] = np.mean(np.sign(output) != y)

        err_exponential[t] = np.mean(np.exp(-y * output))

    return output, err_individual, err_ensemble, err_exponential


def main(argv):
    D = np.loadtxt(argv[1])
    labels = D[:,0].copy()
    D[:,0] = 1

    itercnt = int(argv[2])

    output, err_individual, err_ensemble, err_exponential = adaboost(D, labels, itercnt)

    print('individual error:')
    print(err_individual)
    print('ensemble error:')
    print(err_ensemble)
    print('exponential error:')
    print(err_exponential)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage: python %s filename' % sys.argv[0])
    else:
        main(sys.argv)