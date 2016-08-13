#!/usr/bin/env python
"""
Convolutional variational auto-encoder for MNIST data. The model is
written in TensorFlow, with neural networks using Pretty Tensor.

Probability model
    Prior: Normal
    Likelihood: Bernoulli parameterized by convolutional NN
Variational model
    Likelihood: Mean-field Normal parameterized by convolutional NN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, Normal
from keras import backend as K
from keras.layers import Dense
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow.examples.tutorials.mnist import input_data

#N = ... # num data points
M = 128 # batch size during training
d = 10 # latent variable dimension

# Probability model (subgraph)
def generative_network(z):
    """Output logits for Bernoulli(x | p=sigmoid(logits))."""
    z = tf.identity(z) # necessary for converting distributiontensor to tensor
    z = tf.reshape(z, z.get_shape()[1:]) # TODO remove the (1, ) from sample
    hidden = Dense(64, activation=K.relu)(z) # (M, 64)
    return Dense(28*28)(hidden) # (M, 784)

z = Normal([tf.zeros([M, d]), tf.ones([M, d])])
x = Bernoulli([z], lambda cond_set: generative_network(cond_set[0]))

# Variational model (subgraph)
x_ph = tf.placeholder(tf.float32, [M, 28*28])
hidden = Dense(64, activation='relu')(x_ph) # (M, 64)
loc = Dense(d)(hidden) # (M, d)
scale = Dense(d, activation=K.softplus)(hidden) # (M, d)
qz = Normal([loc, scale])

mnist = input_data.read_data_sets("data/mnist", one_hot=True)
x_ph1 = tf.placeholder(tf.float32, [M, 28*28])
data = {x: x_ph1}

# TODO need kl_multivariate_normal-based VAE inference (or maybe just
# try with reparameterization gradient)
sess = ed.get_session()
K.set_session(sess)
inference = ed.MFVI({z: qz}, data)
inference.initialize()

n_epoch = 100
n_iter_per_epoch = 1000
for epoch in range(n_epoch):
    avg_loss = 0.0

    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
    pbar.start()
    for t in range(n_iter_per_epoch):
        pbar.update(t)
        x_train, _ = mnist.train.next_batch(M)
        _, loss = sess.run([inference.train, inference.loss],
                           feed_dict={x_ph1: x_train, x_ph: x_train})
        avg_loss += loss

    # Take average over all ELBOs during the epoch, and over minibatch
    # of data points (images).
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / N_MINIBATCH

    # Print a lower bound to the average marginal likelihood for an
    # image.
    print("log p(x) >= {:0.3f}".format(avg_loss))
