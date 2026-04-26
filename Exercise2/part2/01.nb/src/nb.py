import sys
import numpy as np


def nb(X, labels):
	"""
	Computes the weight vector w and and bias b corresponding
	to the Naive Bayes classifier with Bernoulli components.

	Parameters
	----------
	X : an array of size (n, k)
		training input data for the classifier, elements must be 0/1
	labels : an array of size n
		training labels for the classifier, elements must be 0/1

	Returns
	-------
	w : an array of size k
		weights corresponding to the classifier
	bias: real number
		bias term corresponding to the classifier
	"""

	cnt, k = X.shape
	w = np.zeros(k)
	b = 0

	# --- ML estimates ---
	# Prior: pi = p(y = 1)
	pi = np.mean(labels)

	# Conditional probabilities: theta[c, d] = p(x_d = 1 | y = c)
	idx1 = (labels == 1)
	idx0 = (labels == 0)

	theta1 = X[idx1].mean(axis=0)   # p(x_d = 1 | y = 1),  shape (k,)
	theta0 = X[idx0].mean(axis=0)   # p(x_d = 1 | y = 0),  shape (k,)

	# Clip to avoid log(0)
	eps = 1e-10
	theta1 = np.clip(theta1, eps, 1 - eps)
	theta0 = np.clip(theta0, eps, 1 - eps)

	# --- Derive linear weights from log-posterior ratio ---
	#
	# Classifier predicts y = 1  iff  log p(y=1|x) > log p(y=0|x)
	#
	# log p(y=1|x) - log p(y=0|x)
	#   = log[pi / (1-pi)]
	#     + sum_d { x_d * log[theta1_d / theta0_d]
	#               + (1 - x_d) * log[(1-theta1_d) / (1-theta0_d)] }
	#
	# Collecting the x_d-dependent part into w_d:
	#   w_d = log(theta1_d / theta0_d) - log((1-theta1_d) / (1-theta0_d))
	#
	# Constant part becomes the bias b:
	#   b   = log(pi / (1-pi)) + sum_d log((1-theta1_d) / (1-theta0_d))

	w = np.log(theta1 / theta0) - np.log((1 - theta1) / (1 - theta0))

	b = (np.log(pi / (1 - pi))
		 + np.sum(np.log((1 - theta1) / (1 - theta0))))

	return w, b


def main(argv):
	D = np.loadtxt(argv[1])
	X = D[:, 1:]
	labels = D[:,0]
	print(nb(X, labels))



# This allows the script to be used as a module and as a standalone program
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('usage: python %s filename' % sys.argv[0])
	else:
		main(sys.argv)