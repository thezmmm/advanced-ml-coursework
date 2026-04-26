import sys
import numpy as np


class Edge:
	def __init__(self, inn, outn):
		self.inn = inn
		self.outn = outn
		self.w = 0


class Neuron:
	def __init__(self):
		self.inedges = []
		self.bias = 0
		self.outedges = []
		self.input = 0
		self.output = 0
		self.delta = 0

	def act(self, x):
		# sigmoid
		return 1.0 / (1.0 + np.exp(-x))

	def der(self, x):
		s = self.act(x)
		return s * (1 - s)

	def compute_output(self):
		self.input = 0.0
		for e in self.inedges:
			self.input += e.w * e.inn.output
		self.input += self.bias
		self.output = self.act(self.input)

	def compute_delta(self):
		self.delta = 0.0
		for e in self.outedges:
			self.delta += e.w * e.outn.delta
		self.delta *= self.der(self.input)


class NN:
	def __init__(self, ncnt):
		self.neurons = [Neuron() for i in range(ncnt)]
		self.edges = []

	def join(self, i, j):
		assert(i < j)
		n1 = self.neurons[i]
		n2 = self.neurons[j]
		edge = Edge(n1, n2)
		n1.outedges.append(edge)
		n2.inedges.append(edge)
		self.edges.append(edge)

	def randomize(self):
		for e in self.edges:
			e.w = np.random.randn()
		for n in self.neurons:
			n.bias = np.random.randn()

	def forward(self, x):
		cnt = len(x)
		for i in range(cnt):
			self.neurons[i].output = x[i]
		for i in range(cnt, len(self.neurons)):
			self.neurons[i].compute_output()

	def backward(self, y):
		n = self.neurons[-1]
		n.delta = -n.der(n.input)
		if y != 1:
			n.delta = -n.delta
		for i in range(len(self.neurons) - 2, -1, -1):
			self.neurons[i].compute_delta()

	def gradients(self, X, y):
		cnt = X.shape[0]
		ncnt = len(self.neurons)
		ecnt = len(self.edges)
		wgrad = np.zeros(ecnt)
		bgrad = np.zeros(ncnt)
		error = 0

		for i in range(cnt):
			self.forward(X[i, :])
			self.backward(y[i])

			error += np.abs(self.neurons[-1].output - y[i]) / cnt

			for j, e in enumerate(self.edges):
				wgrad[j] += e.outn.delta * e.inn.output / cnt

			for j, n in enumerate(self.neurons):
				bgrad[j] += n.delta / cnt

		return wgrad, bgrad, error

	def fit(self, X, y, itercnt, eta):
		for i in range(itercnt):
			wgrad, bgrad, error = self.gradients(X, y)
			print(error)
			for j, e in enumerate(self.edges):
				e.w -= eta * wgrad[j]
			for j, n in enumerate(self.neurons):
				n.bias -= eta * bgrad[j]


def main(argv):
	D = np.loadtxt(argv[1])

	incnt = D.shape[1] - 1
	hiddencnt = int(argv[2])
	itercnt = int(argv[3])
	eta = float(argv[4])

	y = D[:,0]
	X = D[:,1:]

	net = NN(incnt + hiddencnt + 1)

	for i in range(incnt):
		for j in range(hiddencnt):
			net.join(i, incnt + j)

	for i in range(hiddencnt):
		net.join(incnt + i, incnt + hiddencnt)

	np.random.seed(100)
	net.randomize()
	net.fit(X, y, itercnt, eta)


if __name__ == "__main__":
	if len(sys.argv) != 5:
		print('usage: python %s filename hiddencnt itercnt eta' % sys.argv[0])
	else:
		main(sys.argv)