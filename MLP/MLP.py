import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mnist = input_data.read_data_sets('../MNIST',one_hot=True)

# input
x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])

# MLP
def MLP(n,hidden_list,out):
	with tf.contrib.framework.arg_scope([layers.fully_connected],normalizer_fn=layers.batch_norm,normalizer_params={'scale':True}):
		for h in hidden_list:
			n = layers.fully_connected(n,h,activation_fn=tf.nn.relu)
		n = layers.fully_connected(n,out)
	return n

# architecture 784-200-10
pred_logits = MLP(x,[200],10)
pred_labels = tf.argmax(pred_logits,axis=1)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred_logits)
loss = tf.reduce_mean(loss)

global_step = tf.contrib.framework.get_or_create_global_step()
train = layers.optimize_loss(loss,global_step,1e-2,'Adam') 
acc   = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),pred_labels),tf.float32))

batchsize = 50
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for loop in range(50):
		for i in range(mnist.train.images.shape[0]/batchsize):
			images,label = mnist.train.next_batch(batchsize)
			sess.run([train],feed_dict={x:images,y:label})
		test_acc = sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels})
		train_acc = sess.run(acc,feed_dict={x:mnist.train.images,y:mnist.train.labels})
		print 'Loop %d , train acc : %f, test acc : %f' % (loop,train_acc,test_acc)
