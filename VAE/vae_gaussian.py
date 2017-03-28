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
# encoder
W = tf.Variable(xavier_init([p,n_hidden]))
b = tf.Variable(tf.zeros(shape=[n_hidden]))
W2 = tf.Variable(xavier_init([n_hidden,2*n_z]))
b2 = tf.Variable(tf.zeros(shape=[2*n_z]))
def encoder(x):
	h = tf.nn.relu(tf.matmul(x,W) + b)
	para = tf.matmul(h,W2) + b2
	mu = para[:,:n_z]
	sigma = para[:,n_z:]
	return mu,sigma

mu_z,sigma_z = encoder(x)

KL_loss = 0.5 * tf.reduce_sum(mu_z**2 + tf.exp(sigma_z) -1 - sigma_z, axis=1)

sampled_z = mu_z + tf.exp(sigma_z/2) * tf.random_normal(tf.shape(mu_z))

W = tf.Variable(xavier_init([n_z,n_hidden]))
b = tf.Variable(tf.zeros(shape=[n_hidden]))
W2 = tf.Variable(xavier_init([n_hidden,2*p]))
b2 = tf.Variable(tf.zeros(shape=[2*p]))
def decoder(z):
	h = tf.nn.relu(tf.matmul(z,W) + b)
	para = tf.matmul(h,W2) + b2
	mu = para[:,:p]
	sigma = tf.exp(para[:,p:])
	return mu,sigma
def sampleX(z):
	mu,sigma = decoder(z)
	return mu

mu,sigma = decoder(sampled_z)

x_sampled = sampleX(z)

recon_loss = tf.reduce_sum( (x-mu)**2,axis=1)
#recon_loss = tf.reduce_sum( (x-mu)**2 / sigma + tf.log(sigma),axis=1)

vae_loss = tf.reduce_mean(KL_loss + recon_loss)
train_op = tf.train.AdamOptimizer().minimize(vae_loss)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for loop in xrange(100000):
		X,_ = mnist.train.next_batch(64)
		_,loss = sess.run([train_op,vae_loss],feed_dict={x:X})
		if loop % 1000 == 0:
			print 'Loop: %d, Loss: %f' % (loop,loss)

			samples = sess.run(x_sampled,feed_dict={z:np.random.randn(16,n_z)})
			fig = plot(samples)
			plt.savefig('samples/%s.png' % (loop/1000),box_inches='tight')
			plt.close(fig)
