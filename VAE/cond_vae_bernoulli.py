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
c   = mnist.train.labels.shape[1]
print n,p

n_hidden = 128
n_z = 100
# input
x = tf.placeholder(tf.float32,[None,p])
y = tf.placeholder(tf.float32,[None,c])
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
W = tf.Variable(xavier_init([p+c,n_hidden]))
b = tf.Variable(tf.zeros(shape=[n_hidden]))
W2 = tf.Variable(xavier_init([n_hidden,2*n_z]))
b2 = tf.Variable(tf.zeros(shape=[2*n_z]))
def encoder(x,y):
	x = tf.concat([x,y],axis=1)
	h = tf.nn.relu(tf.matmul(x,W) + b)
	para = tf.matmul(h,W2) + b2
	mu = para[:,:n_z]
	sigma = para[:,n_z:]
	return mu,sigma

mu_z,sigma_z = encoder(x,y)

KL_loss = 0.5 * tf.reduce_sum(mu_z**2 + tf.exp(sigma_z) -1 - sigma_z, axis=1)

sampled_z = mu_z + tf.exp(sigma_z/2) * tf.random_normal(tf.shape(mu_z))

W = tf.Variable(xavier_init([n_z+c,n_hidden]))
b = tf.Variable(tf.zeros(shape=[n_hidden]))
W2 = tf.Variable(xavier_init([n_hidden,p]))
b2 = tf.Variable(tf.zeros(shape=[p]))
def decoder(z,y):
	z = tf.concat([z,y],axis=1)
	h = tf.nn.relu(tf.matmul(z,W) + b)
	logits = tf.matmul(h,W2) + b2
	prob = tf.nn.sigmoid(logits)
	return prob,logits
		
prob,logits = decoder(sampled_z,y)
x_sampled,_ = decoder(z,y)

recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=x),axis=1)

vae_loss = tf.reduce_mean(KL_loss + recon_loss)
train_op = tf.train.AdamOptimizer().minimize(vae_loss)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for loop in xrange(100000):
		X,Y = mnist.train.next_batch(64)
		_,loss = sess.run([train_op,vae_loss],feed_dict={x:X,y:Y})
		if loop % 1000 == 0:
			print 'Loop: %d, Loss: %f' % (loop,loss)
			
			Y = np.zeros((16,c))
			Y[:,np.random.randint(0,c)] = 1
			samples = sess.run(x_sampled,feed_dict={z:np.random.randn(16,n_z),y:Y})
			fig = plot(samples)
			plt.savefig('samples/%s.png' % (loop/1000),box_inches='tight')
			plt.close(fig)
