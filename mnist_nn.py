import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave, imresize

sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#fist layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #size: -1*14*14*32

#second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#train
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#evaluation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#run
sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	
writer=tf.summary.FileWriter("tensorflow")
writer.add_graph(sess.graph)

#to print the images
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# x_image = tf.reshape(x, [-1,28,28,1])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1) #size: -1*14*14*32




print(W_conv1.shape)
print(W_conv1)
print(b_conv1.shape)
print(sess.run(b_conv1))
#get the tensor
tensor_conv1=h_conv1.eval(feed_dict={x:mnist.test.images})#[none,748]
#now the tensor is none*14*14*32

#print all the layers
image=np.ones([14,14,3],np.int64)
for i in range(32):
	layer=tensor_conv1[0,:,:,i]
	for a in range(14):
		for b in range(14):
			image[a][b][0]=int(layer[a][b]*255)
			image[a][b][1]=int(layer[a][b]*255)
			image[a][b][2]=int(layer[a][b]*255)
	imsave("./firstlayer/layer"+str(i)+".jpg",layer)

# sample=mnist.test.images[0]

# sample=np.reshape(sample,(28,28,1))

# sample_image=np.ones([28,28,3],np.int64)
# for a in range(28):
	# for b in range(28):
		# sample_image[a][b][0]=int(sample[a][b]*255)
		# sample_image[a][b][1]=int(sample[a][b]*255)
		# sample_image[a][b][2]=int(sample[a][b]*255)

# print(sample_image.shape,sample_image.dtype)
# print(np.reshape(sample_image,[28,28,3]))

# imsave('sample.jpg',sample_image)



























