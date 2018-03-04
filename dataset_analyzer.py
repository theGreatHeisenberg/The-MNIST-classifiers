import numpy as np
import gzip
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import collections
import math
import matplotlib.pyplot as plt


def show_image(data):
	plt.imshow(data, cmap=cm.Greys_r)
	plt.show()

def print_grid(data):
	for row in data:
		for number in row:
			print int(np.sign(number)),
		print "\n"

def get_uncompressed_data(name):
	with gzip.open(name, "rb") as f:
		return pickle.load(f)



def visualize():
	data = get_uncompressed_data("mnist_rowmajor.pkl.gz")
	images_train = data["images_train"]
	labels_train = data["labels_train"] 
	images_test = data["images_test"]
	labels_test = data["labels_test"]
	# data = np.array(images_train[58]).reshape((28,28))
	print labels_train[58]
	# show_image(data)
	# print_grid(data)
	# stratified_sample(labels_train, data)
	test = {"images_test": images_test, "labels_test": labels_test}
	with gzip.open("test.pkl.gz", 'wb') as fp:
		pickle.dump(test, fp, pickle.HIGHEST_PROTOCOL)

def persist(data, name, original_data):
	images_train = []
	labels_train = []
	for item, value in data.items():
		for i in value:
			images_train.append(original_data["images_train"][i])
			labels_train.append(original_data["labels_train"][i])
	sample = {"images_train": np.asarray(images_train), "labels_train": np.asarray(labels_train)}
	with gzip.open(name + ".pkl.gz", 'wb') as fp:
		pickle.dump(sample, fp, pickle.HIGHEST_PROTOCOL)


def analyze(pickle_name, title):
	data = get_uncompressed_data(pickle_name)
	images_train = data["images_train"]
	labels_train = data["labels_train"]
	plt.hist(labels_train, bins=30)
	plt.ylabel("Number of Instances")
	plt.xlabel("Digits")
	plt.title(title)
	plt.show()



def stratified_sample(data, original_data):
	#spliting in 70 int-train, 30 int-dev
	flat_list = data.flatten()
	map_list = {}
	for index, item in enumerate(flat_list):
		if item not in map_list:
			map_list[item] = [index]
		else:
			map_list[item].append(index)
	dev_data = {}
	train_data = {}
	for item, values in map_list.items():
		train_data[item] = [values[i] for i in range(0, int(math.floor(len(values)*0.7))+1)]
		dev_data[item] = [values[i] for i in range(int(math.ceil(len(values)*0.7)), len(values))]
	persist(dev_data, "int-dev", original_data)
	persist(train_data, "int-train", original_data)

# visualize()
# analyze("int-train.pkl.gz", "Histogram - int-train data (70% of original training data)")
# analyze("int-dev.pkl.gz", "Histogram - int-dev data (30% of original training data)")
# analyze("mnist_rowmajor.pkl.gz")