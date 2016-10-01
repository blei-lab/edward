#!/usr/bin/env python
"""
Variational auto-encoder for MNIST data.
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

M = 100  # batch size during training
d = 2  # latent variable dimension
ed.set_seed(42)

# Probability model (subgraph)
z = Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
hidden = Dense(256, activation=K.relu)(z.value())
x = Bernoulli(logits=Dense(28 * 28)(hidden))

# Variational model (subgraph)
x_ph = ed.placeholder(tf.float32, [M, 28 * 28])
hidden = Dense(256, activation=K.relu)(x_ph)
qz = Normal(mu=Dense(d)(hidden),
            sigma=Dense(d, activation=K.softplus)(hidden))

# Bind p(x, z) and q(z | x) to the same TensorFlow placeholder for x.
mnist = input_data.read_data_sets("data/mnist", one_hot=True)
data = {x: x_ph}

sess = ed.get_session()
K.set_session(sess)
inference = ed.MFVI({z: qz}, data)
optimizer = tf.train.RMSPropOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)

init = tf.initialize_all_variables()
init.run()

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
        info_dict = inference.update(feed_dict={x_ph: x_train})
        avg_loss += info_dict['loss']

    # Take average over all ELBOs during the epoch, and over minibatch
    # of data points (images).
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / M

    # Print a lower bound to the average marginal likelihood for an
    # image.
    print("log p(x) >= {:0.3f}".format(avg_loss))

    # Prior predictive check.
    imgs = sess.run(x.value())
    for m in range(M):
        imsave("img/%d.png" % m, imgs[m].reshape(28, 28))
