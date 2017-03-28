import tensorflow as tf
import numpy as np
import os
import matplotlib
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.system('rm samples/*')

# loading mnist data
mnist = input_data.read_data_sets('../MNIST',one_hot=True)
n,p = mnist.train.images.shape
print n,p

n_hidden = 128
n_z = 100
# input
x = tf.placeholder(tf.float32,[None,p])
z = tf.placeholder(tf.float32,[None,n_z])

def plot(samples):
	fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
	return fig

def xavier_init(size):
	return tf.random_normal(shape=size,stddev= 1. / tf.sqrt(size[0]/2.))
def concat_elu(inputs):
    return tf.nn.elu(tf.concat(3, [-inputs, inputs]))

# discriminator
def discriminator(x):
	x = tf.reshape(x,[-1,28,28,1])
	x = layers.conv2d(x,32,5,stride=2)
	x = layers.conv2d(x,64,5,stride=2)
	x = layers.conv2d(x,128,5,padding='VALID')
	x = layers.flatten(x)
	logit = layers.fully_connected(x,1,activation_fn=None)
	prob  = tf.nn.sigmoid(logit)
	return prob,logit

# generator
def generator(z):
	z = tf.expand_dims(z,1)
	n = tf.expand_dims(z,1)
	n = layers.conv2d_transpose(n,128,3,padding='VALID')
	n = layers.conv2d_transpose(n,64,5,padding='VALID')
	n = layers.conv2d_transpose(n,32,5,stride=2)
	n = layers.conv2d_transpose(n,1,5,stride=2,activation_fn=tf.nn.sigmoid)
	x = layers.flatten(n)
	return x

with tf.contrib.framework.arg_scope([layers.conv2d,layers.conv2d_transpose],activation_fn=tf.nn.elu,normalizer_fn=layers.batch_norm,normalizer_params={'scale':True}):
	with tf.variable_scope('model'):
		prob_real,logit_real = discriminator(x)
		D_params_num = len(tf.trainable_variables())

		sampled_x = generator(z)
	with tf.variable_scope('model',reuse=True):
		prob_fake,logit_fake = discriminator(sampled_x)

para_D = tf.trainable_variables()[:D_params_num]
para_G = tf.trainable_variables()[D_params_num:]
D_loss = -tf.reduce_mean(tf.log(prob_real) + tf.log(1. - prob_fake))
G_loss = -tf.reduce_mean(tf.log(prob_fake))

D_train = tf.train.AdamOptimizer(1e-3).minimize(D_loss,var_list=para_D)
G_train = tf.train.AdamOptimizer(1e-2).minimize(G_loss,var_list=para_G)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for loop in xrange(100000):
		X,_ = mnist.train.next_batch(64)
		
		_,_D_loss = sess.run([D_train,D_loss],feed_dict={x:X,z:np.random.uniform(-1,1,size=[64,n_z])})
		_,_G_loss = sess.run([G_train,G_loss],feed_dict={z:np.random.uniform(-1,1,size=[64,n_z])})

		if loop % 1000 == 0:
			print 'Loop: %d, D Loss: %f, G Loss : %f' % (loop,_D_loss,_G_loss)

			samples = sess.run(sampled_x,feed_dict={z:np.random.uniform(-1,1,size=[16,n_z])})
			fig = plot(samples)
			plt.savefig('samples/%s.png' % (loop/1000),box_inches='tight')
			plt.close(fig)
