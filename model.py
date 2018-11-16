import tensorflow as tf
import numpy as np

class MnistSumDataset():
	def __init__(self):
		self.dataset = tf.contrib.learn.datasets.mnist.load_mnist()
		self.train_labels = self.dataset.train.labels
		self.train_images = self.dataset.train.images
		self.labels_set = set(range(10))
		self.label_to_indices = {label: np.where(self.train_labels == label)[0] for label in self.labels_set}

	def __getitem__(self, index):
		a, ai = self.train_images[index], self.train_labels[index]
		bi = np.random.choice(list(self.labels_set ))
		b = np.random.choice(self.label_to_indices[bi])
		b = self.train_images[b]
		ci = ai + bi
		ten = 10 if ci >= 10 else 0
		ci = ci - ten
		c = np.random.choice(self.label_to_indices[ci])
		c = self.train_images[c]
		return np.stack((a, b, c)), np.array([ten], dtype=np.int32), np.stack((ai, bi, ci))


	def __len__(self):
		return len(self.train_labels)

	def generator(self):
		for i in range(self.__len__()):
			yield self.__getitem__(i)

def build_model():
	model = tf.keras.models.Sequential([
	  tf.keras.layers.InputLayer(input_shape=(784,)),
	  tf.keras.layers.Reshape((28, 28, 1)),
	  tf.keras.layers.Conv2D(20, 5, activation=tf.nn.relu),
	  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
	  tf.keras.layers.Conv2D(50, 5, activation=tf.nn.relu),
	  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
	  tf.keras.layers.Flatten(),
	  tf.keras.layers.Dense(512, activation=tf.nn.relu),
	  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	return model

def build_graph():
	model = build_model()
	BATCH_SIZE = 128
	d = MnistSumDataset()

	dataset = tf.data.Dataset.from_generator(d.generator, output_types=(tf.float32, tf.float32, tf.int32))
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()

	images, tens, labels = iterator.get_next()

	a, b, c = tf.split(images, 3, axis=1)
	ai, bi, ci = tf.split(labels, 3, axis=1)

	ad = model(a)
	bd = model(b)
	cd = model(c)
	W1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype = 'float32', shape = [10, 1])

	y1 = tf.matmul(ad, W1)
	y2 = tf.matmul(bd, W1)
	y3 = tf.matmul(cd, W1)

	pro = tf.reduce_mean(ad, axis = 0)
	_, var = tf.nn.moments(pro - 0.1, 0)

	loss1 = tf.losses.absolute_difference(y1+y2, y3+tens)
	loss2 = 15*var
	loss = loss1 + loss2
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.001
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           	300, 0.6, staircase=True)

	optimizor = tf.train.AdamOptimizer(learning_rate=learning_rate)

	train_op = optimizor.minimize(loss, global_step=global_step)
	return y1, y2, y3, ai, bi, ci, train_op, loss




