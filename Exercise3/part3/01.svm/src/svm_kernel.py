import sys
import numpy as np


class SVM:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel

    def step(self, i, j):
        a1 = self.alpha[i]
        a2 = self.alpha[j]
        y1 = self.y[i]
        y2 = self.y[j]
        E1 = self.u[i] - y1
        E2 = self.u[j] - y2

        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0, a1 + a2 - self.C)
            H = min(self.C, a1 + a2)

        if L == H:
            return False

        eta = (self.kernel(self.X[i], self.X[i]) +
               self.kernel(self.X[j], self.X[j]) -
               2 * self.kernel(self.X[i], self.X[j]))

        if eta > 0:
            n2 = a2 + y2 * (E1 - E2) / eta
            n2 = min(H, max(L, n2))
        else:
            if y2 * (E1 - E2) < -10e-6:
                n2 = L
            elif y2 * (E1 - E2) > 10e-6:
                n2 = H
            else:
                return False

        if abs(n2 - a2) < 10e-6:
            return False

        n1 = a1 - y1 * y2 * (n2 - a2)
        n1 = min(self.C, max(0, n1))

        self.alpha[i] = n1
        self.alpha[j] = n2

        d1 = n1 - a1
        d2 = n2 - a2
        for n in range(len(self.y)):
            self.u[n] += (d1 * y1 * self.kernel(self.X[i], self.X[n]) +
                          d2 * y2 * self.kernel(self.X[j], self.X[n]))

        return True

    def optimize(self):
        cnt = self.X.shape[0]
        changes = True
        round = 0
        giveup = 1000

        while changes and round < giveup:
            changes = False
            round += 1
            for i in range(cnt):
                for j in range(cnt):
                    if self.step(i, j):
                        changes = True

        for i in range(cnt):
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                self.b = self.y[i] - self.u[i]
                break

    def fit(self, X, y):
        cnt = X.shape[0]
        self.X = X
        self.y = y
        self.u = np.zeros(cnt)
        self.alpha = np.zeros(cnt)
        self.b = 0
        self.optimize()

    def score(self, X):
        """
        score(z) = sum_j alpha[j] * y[j] * K(x[j], z) + b
        """
        n_test = X.shape[0]
        scores = np.zeros(n_test)
        for idx in range(n_test):
            s = 0.0
            for j in range(len(self.y)):
                s += self.alpha[j] * self.y[j] * self.kernel(self.X[j], X[idx])
            scores[idx] = s + self.b
        return scores

    def predict(self, X):
        return np.sign(self.score(X))


def linear_kernel(x, y):
    return x @ y


def rbf_kernel(x, y):
    return np.exp(-0.5 * np.sum((x - y) ** 2))


def main(argv):
    D = np.loadtxt(argv[1])
    y = D[:, 0]
    X = D[:, 1:]

    penalty = float(argv[2])

    svm = SVM(penalty, rbf_kernel)
    svm.fit(X, y)

    print('predictions for training data:')
    print(svm.predict(X))

    print('training error:')
    print(np.mean(svm.predict(X) != y))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage: python %s filename penalty' % sys.argv[0])
    else:
        main(sys.argv)