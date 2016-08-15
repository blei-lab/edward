#!/usr/bin/env python
"""
Convolutional variational auto-encoder for MNIST data.
Assumes the directories "img/" and "data/mnist/" exist.
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
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

M = 128 # batch size during training
d = 10 # latent variable dimension
ed.set_seed(42)

# Probability model (subgraph)
z = Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
hidden = Dense(64)(tf.identity(z)) # (M, 64); identity() for tensor conversion
logits = Dense(28*28)(hidden)
x = Bernoulli(logits=logits) # (M, 784)

# Variational model (subgraph)
x_ph = tf.placeholder(tf.float32, [M, 28*28])
tf.add_to_collection('placeholders', x_ph)
hidden = Dense(64, activation=K.sigmoid)(x_ph) # (M, 64)
mu = Dense(d)(hidden) # (M, d)
sigma = Dense(d, activation=K.softplus)(hidden) # (M, d)
qz = Normal(mu=mu, sigma=sigma)

# Bind p(x, z) and q(z | x) to the same TensorFlow placeholder for x.
mnist = input_data.read_data_sets("data/mnist", one_hot=True)
data = {x: x_ph}

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
                           feed_dict={x_ph: x_train})
        avg_loss += loss

    # Take average over all ELBOs during the epoch, and over minibatch
    # of data points (images).
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / M

    # Print a lower bound to the average marginal likelihood for an
    # image.
    print("log p(x) >= {:0.3f}".format(avg_loss))

    # Prior predictive check.
    imgs = sess.run(x.value())
    for b in range(M):
        imsave("img/%d.png" % b, imgs[b].reshape(28, 28))
