import sys
from numpy import array, loadtxt, argsort
from numpy.linalg import eig


def covariance_matrix(X, bias=False):
    """
    Computes covariance matrix.

    Parameters
    ----------
    X : an array of size (n, k)
       input data
    bias: bool, optional
       If True, then the normalization should be n, otherwise it is n - 1.
       Default value is False.

    Returns
    -------
    C : an array of size (k, k)
       covariance matrix
    """

    n = X.shape[0]
    mu = X.mean(axis=0)          # shape (k,)
    X_centered = X - mu          # shape (n, k)
    C = X_centered.T @ X_centered  # shape (k, k)

    if bias:
        return C / n
    else:
        return C / (n - 1)


def pca(X):
    """
    Computes PCA with 2 components

    Parameters
    ----------
    X : an array of size (n, k)
       input data

    Returns
    -------
    v1 : an array of size n
       ith element = first principal component of the ith data point
    v2 : an array of size n
       ith element = second principal component of the ith data point
    """

    C = covariance_matrix(X, bias=False)      # (k, k)
    eigenvalues, eigenvectors = eig(C)        # eigenvalues: (k,), eigenvectors: (k, k)

    # eig may return complex values due to numerical noise; take real part
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort by descending eigenvalue; pick top 2
    idx = argsort(eigenvalues)[::-1]
    w1 = eigenvectors[:, idx[0]]   # first principal eigenvector, shape (k,)
    w2 = eigenvectors[:, idx[1]]   # second principal eigenvector, shape (k,)

    # Project data onto eigenvectors
    v1 = X @ w1    # shape (n,)
    v2 = X @ w2    # shape (n,)

    return v1, v2


def main(argv):
    X = loadtxt(argv[1])
    print(covariance_matrix(X))
    print(pca(X))



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__":
    if len(sys.argv) != 2:
       print('usage: python %s filename' % sys.argv[0])
    else:
       main(sys.argv)