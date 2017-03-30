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

n_hidden = 100
n_z = 10
# input
x = tf.placeholder(tf.float32,[None,p])
z = tf.placeholder(tf.float32,[None,n_z])
z2 = tf.placeholder(tf.float32,[None,n_z])

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

encoder_params_num = len(tf.trainable_variables())

mu_z,sigma_z = encoder(x)

sampled_z = mu_z + tf.exp(sigma_z/2) * tf.random_normal(tf.shape(mu_z))

W = tf.Variable(xavier_init([n_z,n_hidden]))
b = tf.Variable(tf.zeros(shape=[n_hidden]))
W2 = tf.Variable(xavier_init([n_hidden,p]))
b2 = tf.Variable(tf.zeros(shape=[p]))
def decoder(z):
	h = tf.nn.relu(tf.matmul(z,W) + b)
	logits = tf.matmul(h,W2) + b2
	prob = tf.nn.sigmoid(logits)
	return prob,logits
decoder_params_num = len(tf.trainable_variables()) - encoder_params_num

prob,logits = decoder(sampled_z)
x_sampled,_ = decoder(z)

recon_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=x),axis=1))

W = tf.Variable(xavier_init([n_z,n_hidden]))
b = tf.Variable(tf.zeros(shape=[n_hidden]))
W2 = tf.Variable(xavier_init([n_hidden,1]))
b2 = tf.Variable(tf.zeros(shape=[1]))
def discriminator(z):
	h = tf.nn.relu(tf.matmul(z,W) + b)
	logits = tf.matmul(h,W2) + b2
	prob = tf.nn.sigmoid(logits)
	return prob,logits
discriminator_params_num = len(tf.trainable_variables()) - encoder_params_num - decoder_params_num

encoder_params = tf.trainable_variables()[:encoder_params_num]
decoder_params = tf.trainable_variables()[encoder_params_num:(encoder_params_num+decoder_params_num)]
discrminator_params = tf.trainable_variables()[(encoder_params_num+decoder_params_num):]

prob_fake,logit_fake = discriminator(sampled_z)
prob_real,logit_real = discriminator(z2)


discrmin_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real,labels=tf.ones_like(logit_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake,labels=tf.zeros_like(logit_fake)))
generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake,labels=tf.ones_like(logit_fake)))

train_encoder_decoder = tf.train.AdamOptimizer(1e-4).minimize(recon_loss,var_list=encoder_params.extend(decoder_params))
train_discriminator   = tf.train.AdamOptimizer(1e-4).minimize(discrmin_loss,var_list=discrminator_params)
train_generator       = tf.train.AdamOptimizer(1e-4).minimize(generator_loss,var_list=encoder_params)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for loop in xrange(100000):
		X,_ = mnist.train.next_batch(64)
		Z = np.random.randn(64,n_z)

		loss1,_ = sess.run([recon_loss,train_encoder_decoder],feed_dict={x:X})
		loss2,_ = sess.run([discrmin_loss,train_discriminator],feed_dict={x:X,z2:Z})
		loss3,_ = sess.run([generator_loss,train_generator],feed_dict={x:X})
		if loop % 1000 == 0:
			print 'Loop: %d, Loss: %f %f %f' % (loop,loss1,loss2,loss3)

			samples = sess.run(x_sampled,feed_dict={z:np.random.randn(16,n_z)})
			fig = plot(samples)
			plt.savefig('samples/%s.png' % (loop/1000),box_inches='tight')
			plt.close(fig)
