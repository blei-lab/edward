#!/usr/bin/env python
"""Wasserstein generative adversarial network for MNIST (Arjovsky et
al., 2017). It modifies GANs (Goodfellow et al., 2014) to optimize
under the Wasserstein distance.

References
----------
http://edwardlib.org/tutorials/gan
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from edward.models import Uniform
from observations import mnist
from tensorflow.contrib import slim


def generator(array, batch_size):
  """Generate batch with respect to array's first axis."""
  start = 0  # pointer to where we are in iteration
  while True:
    stop = start + batch_size
    diff = stop - array.shape[0]
    if diff <= 0:
      batch = array[start:stop]
      start += batch_size
    else:
      batch = np.concatenate((array[start:], array[:diff]))
      start = diff
    batch = batch.astype(np.float32) / 255.0  # normalize pixel intensities
    batch = np.random.binomial(1, batch)  # binarize images
    yield batch


def generative_network(eps):
  h1 = slim.fully_connected(eps, 128, activation_fn=tf.nn.relu)
  x = slim.fully_connected(h1, 784, activation_fn=tf.sigmoid)
  return x


def discriminative_network(x):
  h1 = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
  h2 = slim.fully_connected(h1, 1, activation_fn=None)
  return h2


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


ed.set_seed(42)

data_dir = "/tmp/data"
out_dir = "/tmp/out"
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
M = 128  # batch size during training
d = 10  # latent dimension

# DATA. MNIST batches are fed at training time.
(x_train, _), (x_test, _) = mnist(data_dir)
x_train_generator = generator(x_train, M)
x_ph = tf.placeholder(tf.float32, [M, 784])

# MODEL
with tf.variable_scope("Gen"):
  eps = Uniform(low=tf.zeros([M, d]) - 1.0, high=tf.ones([M, d]))
  x = generative_network(eps)

# INFERENCE
optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
optimizer_d = tf.train.RMSPropOptimizer(learning_rate=5e-5)

inference = ed.WGANInference(
    data={x: x_ph}, discriminator=discriminative_network)
inference.initialize(
    optimizer=optimizer, optimizer_d=optimizer_d,
    n_iter=15000, n_print=1000, clip=0.01, penalty=None)

sess = ed.get_session()
tf.global_variables_initializer().run()

idx = np.random.randint(M, size=16)
i = 0
for t in range(inference.n_iter):
  if t % inference.n_print == 0:
    samples = sess.run(x)
    samples = samples[idx, ]

    fig = plot(samples)
    plt.savefig(os.path.join(out_dir, '{}.png').format(
        str(i).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    i += 1

  x_batch = next(x_train_generator)
  for _ in range(5):
    inference.update(feed_dict={x_ph: x_batch}, variables="Disc")

  info_dict = inference.update(feed_dict={x_ph: x_batch}, variables="Gen")
  # note: not printing discriminative objective; ``info_dict`` above
  # does not store it since updating only "Gen"
  info_dict['t'] = info_dict['t'] // 6  # say set of 6 updates is 1 iteration
  inference.print_progress(info_dict)
