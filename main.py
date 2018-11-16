import tensorflow as tf
import numpy as np
from model import build_graph
from model import build_model
import os
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 
                           '1', 
                           '0 or 1'
                           )

def get_accuracy(s1, s2, s3, s4, s5, s6, BATCH_SIZE):
	predictions1 = np.round(s1)
	predictions2 = np.round(s2)
	predictions3 = np.round(s3)
	accuracy = (np.sum(predictions1==s4)+np.sum(predictions2==s5)+np.sum(predictions3==s6))/BATCH_SIZE/3
	return accuracy


def train():
	BATCH_SIZE = 128
	y1, y2, y3, ai, bi, ci, train_op, loss = build_graph()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# saver.restore(sess, "./tmp/model.ckpt")
		for i in range(5000):
			_, s1, s2, s3, s4, s5, s6, s7 = sess.run([train_op, y1, y2, y3, ai, bi, ci, loss])
			if i % 100 == 0:
				accuracy = get_accuracy(s1, s2, s3, s4, s5, s6, BATCH_SIZE)
				print('Step %s: Accuracy of this batch is %s%%. Loss: %s '%(i, accuracy*100, s7))
				print('The first number: ', np.round(s1[:10, 0]))
				print('The second number:', np.round(s2[:10, 0]))
				print('The third number: ', np.round(s3[:10, 0]))
			# if i > 1000:
			# 	accuracy = get_accuracy(s1, s2, s3, s4, s5, s6, BATCH_SIZE)
			# 	if(accuracy == 1 and s7<0.06): 
			# 			break
		checkpoint_path = os.path.join('./tmp', 'model.ckpt')
		saver.save(sess, checkpoint_path)

def eval():
	dataset = tf.contrib.learn.datasets.mnist.load_mnist()
	test_images = dataset.test.images
	test_labels = dataset.test.labels
	model = build_model()
	aa = model(test_images)
	W1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype = 'float32', shape = [10, 1])
	y1 = tf.matmul(aa, W1)
	# y1 = tf.argmax(aa, axis=1)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "./tmp/model.ckpt")
		s1 = sess.run([y1])
		predictions = np.round(s1)
		accuracy = np.sum(predictions.reshape(10000)==test_labels)/len(test_labels)
		print('Accuracy of the test images: %s%%'%(accuracy*100))
	test_images = 255*np.array(test_images)
	img = 255-test_images[0].reshape(28,28)
	for i in range(1, 10):
		img = np.hstack((img, 255-255*test_images[i].reshape(28,28)))
	img = cv2.resize(img, (1400, 140))
	img = np.vstack((img, 255*np.ones((100,1400)))) 
	for i in range(10):
		cv2.putText(img, '%.2f'%s1[0][i], (i*140+25, 200), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,0,0), thickness=3)
	cv2.imwrite('a.jpg', img)




def main():
	if(FLAGS.mode == '0'):
		train()
	elif(FLAGS.mode == '1'):
		eval()

if __name__ == '__main__':
  main()