#import
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave, imresize
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#build the graph

#the input placeholder
image_vector=tf.placeholder("float",shape=[None,784])
labels=tf.placeholder("float",shape=[None,10])

#first layer
weight_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
bias_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
images=tf.reshape(image_vector,[-1,28,28,1])
result_conv1=tf.nn.conv2d(images,weight_conv1,strides=[1,1,1,1],padding="SAME")+bias_conv1
nonlinear_conv1=tf.nn.relu(result_conv1)
pooling_conv1=tf.nn.max_pool(nonlinear_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#second layer
weight_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
bias_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
result_conv2=tf.nn.conv2d(pooling_conv1,weight_conv2,strides=[1,1,1,1],padding="SAME")+bias_conv2
nonlinear_conv2=tf.nn.relu(result_conv2)
pooling_conv2=tf.nn.max_pool(nonlinear_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#dense layer
pooling_conv2_linear=tf.reshape(pooling_conv2,[-1,7*7*64])
weight_dense=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
bias_dense=tf.Variable(tf.constant(0.1,shape=[1024]))
result_dense=tf.matmul(pooling_conv2_linear,weight_dense)+bias_dense
nonlinear_dense=tf.nn.relu(result_dense)

#add the drop out layer here!!!!!!!!!!!!
keep_prob=tf.placeholder("float")
dropout_dense=tf.nn.dropout(nonlinear_dense,keep_prob)

#output layer
weight_output=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
bias_output=tf.Variable(tf.constant(0,1,shape=[10]))
result_output=tf.matmul(dropout_dense,weight_output)+bias_output
softmax_output=tf.nn.softmax(result_output)

#calculate the loss
cross_entropy=-tf.reduce_sum(labels*tf.log(softmax_output))
#train
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#evaluation
correct_prediction=tf.equal(tf.argmax(softmax_output,1),tf.argmax(labels,1))# there should be one dimension indicator
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


#run
sess=tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
	batch=mnist.train.next_batch(50)
	sess.run(train_step,feed_dict={image_vector:batch[0],labels:batch[1],keep_prob:0.5})
	if i%100==0:
		train_accuracy=sess.run(accuracy, feed_dict={image_vector:batch[0],labels:batch[1],keep_prob:1.0})
		print ("step %d, training accuracy %g"%(i,train_accuracy))
	
#print the total accuracy after training
total_accuracy=sess.run(accuracy,feed_dict={image_vector: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
print ("total accuracy:%g"%(total_accuracy)) # why I need to use %g in stead of %d??????????????