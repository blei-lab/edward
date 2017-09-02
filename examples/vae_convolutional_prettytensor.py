#!/usr/bin/env python
"""Convolutional variational auto-encoder for binarized MNIST.

The neural networks are written with Pretty Tensor.

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
import prettytensor as pt
import tensorflow as tf

from vae_convolutional_util import deconv2d
from edward.models import Bernoulli, Normal
from edward.util import Progbar
from observations import mnist
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data


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


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  with pt.defaults_scope(activation_fn=tf.nn.elu,
                         batch_normalize=True,
                         scale_after_normalization=True):
    return (pt.wrap(z).
            reshape([M, 1, 1, d]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=None).
            flatten()).tensor


def inference_network(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  loc, scale = neural_network(x)
  """
  with pt.defaults_scope(activation_fn=tf.nn.elu,
                         batch_normalize=True,
                         scale_after_normalization=True):
    params = (pt.wrap(x).
              reshape([M, 28, 28, 1]).
              conv2d(5, 32, stride=2).
              conv2d(5, 64, stride=2).
              conv2d(5, 128, edges='VALID').
              dropout(0.9).
              flatten().
              fully_connected(d * 2, activation_fn=None)).tensor

  loc = params[:, :d]
  scale = tf.nn.softplus(params[:, d:])
  return loc, scale


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

# MODEL
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
logits = generative_network(z)
x = Bernoulli(logits=logits)

# INFERENCE
x_ph = tf.placeholder(tf.int32, [M, 28 * 28])
loc, scale = inference_network(tf.cast(x_ph, tf.float32))
qz = Normal(loc=loc, scale=scale)

# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.ReparameterizationKLKLqp({z: qz}, data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer, use_prettytensor=True)

hidden_rep = tf.sigmoid(logits)

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

  # Visualize hidden representations.
  images = hidden_rep.eval()
  for m in range(M):
    imsave(os.path.join(out_dir, '%d.png') % m, images[m].reshape(28, 28))
