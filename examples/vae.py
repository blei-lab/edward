#!/usr/bin/env python
"""Variational auto-encoder for MNIST data.

References
----------
http://edwardlib.org/tutorials/decoder
http://edwardlib.org/tutorials/inference-networks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import Progbar
from keras.layers import Dense
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

ed.set_seed(42)

M = 100  # batch size during training
d = 2  # latent dimension

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# MODEL
# Define a subgraph of the full model, corresponding to a minibatch of
# size M.
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
hidden = Dense(256, activation='relu')(z)
x = Bernoulli(logits=Dense(28 * 28)(hidden))

# INFERENCE
# Define a subgraph of the variational model, corresponding to a
# minibatch of size M.
x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
hidden = Dense(256, activation='relu')(tf.cast(x_ph, tf.float32))
qz = Normal(loc=Dense(d)(hidden),
            scale=Dense(d, activation='softplus')(hidden))

# Bind p(x, z) and q(z | x) to the same TensorFlow placeholder for x.
inference = ed.KLqp({z: qz}, data={x: x_ph})
optimizer = tf.train.RMSPropOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()

n_epoch = 100
n_iter_per_epoch = 1000
for epoch in range(n_epoch):
  avg_loss = 0.0

  pbar = Progbar(n_iter_per_epoch)
  for t in range(1, n_iter_per_epoch + 1):
    pbar.update(t)
    x_train, _ = mnist.train.next_batch(M)
    x_train = np.random.binomial(1, x_train)
    info_dict = inference.update(feed_dict={x_ph: x_train})
    avg_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / M
  print("log p(x) >= {:0.3f}".format(avg_loss))

  # Prior predictive check.
  imgs = sess.run(x)
  for m in range(M):
    imsave(os.path.join(IMG_DIR, '%d.png') % m, imgs[m].reshape(28, 28))
