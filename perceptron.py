import numpy as np
import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import collections
import math
import matplotlib.pyplot as plt

class Perceptron(object):
	def __init__(self, train_data, dev_data, test_data):
		self.train_data = train_data
		self.weights = np.zeros((10, 784))
		self.data = self.get_uncompressed_data(train_data)
		self.dev_data = self.get_uncompressed_data(dev_data)
		self.test_data = self.get_uncompressed_data(test_data)

	def get_uncompressed_data(self, name):
		with gzip.open(name, "rb") as f:
			return pickle.load(f)

	def predict(self, x):
		# print np.argmax(np.matmul(x, np.transpose(self.weights)))
		return np.argmax(np.matmul(x, np.transpose(self.weights)))

	def update(self, x, y, yhat):
		self.weights[y]+= x
		self.weights[yhat]-= x

	def train(self):
		for iteration in range(0,50):
			correct_count = 0
			for index in range(0, len(self.data["images_train"])):
				x,y = self.data["images_train"][index], self.data["labels_train"][index][0]
				# print x,y
				yhat = self.predict(x)
				# print y
				if y == yhat:
					correct_count+=1
				else:
					self.update(x,y,yhat)
			print "Training accuracy at iteration: " + str(iteration) + " is ", 100 * correct_count/float(len(self.data["labels_train"]))

	def dev(self):
		for iteration in range(0,1):
			correct_count = 0
			for index in range(0, len(self.dev_data["images_train"])):
				x,y = self.dev_data["images_train"][index], self.dev_data["labels_train"][index][0]
				# print x,y
				yhat = self.predict(x)
				# print y
				if y == yhat:
					correct_count+=1
			print "correct count: ", correct_count, len(self.dev_data["labels_train"])
			print "dev accuracy at iteration: " + str(iteration) + " is ", 100 * correct_count/float(len(self.dev_data["labels_train"]))

	def test(self):
		for iteration in range(0,1):
			correct_count = 0
			for index in range(0, len(self.test_data["images_test"])):
				x,y = self.test_data["images_test"][index], self.test_data["labels_test"][index][0]
				# print x,y
				yhat = self.predict(x)
				# print y
				if y == yhat:
					correct_count+=1
			print "correct count: ", correct_count, len(self.test_data["labels_test"])
			print "test accuracy at iteration: " + str(iteration) + " is ", 100 * correct_count/float(len(self.test_data["labels_test"]))


p = Perceptron("../int-train.pkl.gz", "../int-dev.pkl.gz", "../test.pkl.gz")
p.train()
p.dev()
p.test()