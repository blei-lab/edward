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
from observations import mnist
from scipy.misc import imsave


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


ed.set_seed(42)

data_dir = "/tmp/data"
out_dir = "/tmp/out"
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
M = 100  # batch size during training
d = 2  # latent dimension

# DATA. MNIST batches are fed at training time.
(x_train, _), (x_test, _) = mnist(data_dir)
x_train_generator = generator(x_train, M)

# MODEL
# Define a subgraph of the full model, corresponding to a minibatch of
# size M.
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
hidden = Dense(256, activation='relu')(z.value())
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

tf.global_variables_initializer().run()

n_epoch = 100
n_iter_per_epoch = x_train.shape[0] // M
for epoch in range(1, n_epoch + 1):
  print("Epoch: {0}".format(epoch))
  avg_loss = 0.0

  pbar = Progbar(n_iter_per_epoch)
  for t in range(1, n_iter_per_epoch + 1):
    pbar.update(t)
    x_batch = next(x_train_generator)
    info_dict = inference.update(feed_dict={x_ph: x_batch})
    avg_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / M
  print("-log p(x) <= {:0.3f}".format(avg_loss))

  # Prior predictive check.
  images = x.eval()
  for m in range(M):
    imsave(os.path.join(out_dir, '%d.png') % m, images[m].reshape(28, 28))
