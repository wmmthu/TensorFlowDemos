import tensorflow as tf
import numpy as np
import os
import matplotlib
from tensorflow.examples.tutorials.mnist import input_data
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

# discriminator
D_W = tf.Variable(xavier_init([p,n_hidden]))
D_b = tf.Variable(tf.zeros(shape=[n_hidden]))
D_W2 = tf.Variable(xavier_init([n_hidden,1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
def discriminator(x):
	h = tf.nn.relu(tf.matmul(x,D_W) + D_b)
	logit = tf.matmul(h,D_W2) + D_b2
	prob  = tf.nn.sigmoid(logit)
	return prob,logit
para_D = [D_W,D_b,D_W2,D_b2]
# generator
G_W = tf.Variable(xavier_init([n_z,n_hidden]))
G_b = tf.Variable(tf.zeros(shape=[n_hidden]))
G_W2 = tf.Variable(xavier_init([n_hidden,p]))
G_b2 = tf.Variable(tf.zeros(shape=[p]))
def generator(z):
	h = tf.nn.relu(tf.matmul(z,G_W) + G_b)
	x = tf.nn.sigmoid(tf.matmul(h,G_W2) + G_b2)
	return x
para_G = [G_W,G_b,G_W2,G_b2]
sampled_x = generator(z)

prob_real,logit_real = discriminator(x)
prob_fake,logit_fake = discriminator(sampled_x)

D_loss = -tf.reduce_mean(tf.log(prob_real) + tf.log(1. - prob_fake))
G_loss = -tf.reduce_mean(tf.log(prob_fake))

D_train = tf.train.AdamOptimizer().minimize(D_loss,var_list=para_D)
G_train = tf.train.AdamOptimizer().minimize(G_loss,var_list=para_G)

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
