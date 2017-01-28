#!/usr/bin/env python
"""Convolutional variational auto-encoder for binarized MNIST.

The model is written in TensorFlow. The neural networks are written
with Pretty Tensor.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import os
import prettytensor as pt
import tensorflow as tf

from tf_convolutional_vae_util import deconv2d
from edward.models import Normal
from edward.stats import bernoulli
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data


class NormalBernoulli:
  """Each binarized pixel in an image is modeled by a Bernoulli
  likelihood. The success probability for each pixel is the output
  of a neural network that takes samples from a normal prior as
  input.

  p(x, z) = Bernoulli(x | logits = neural_network(z)) Normal(z; 0, I)
  """
  def __init__(self, n_vars):
    self.n_vars = n_vars  # number of local latent variables

  def generative_network(self, z):
    """Generative network to parameterize generative model. It takes
    latent variables as input and outputs the likelihood parameters.

    logits = neural_network(z)
    """
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           scale_after_normalization=True):
      return (pt.wrap(z).
              reshape([M, 1, 1, self.n_vars]).
              deconv2d(3, 128, edges='VALID').
              deconv2d(5, 64, edges='VALID').
              deconv2d(5, 32, stride=2).
              deconv2d(5, 1, stride=2, activation_fn=None).
              flatten()).tensor

  def log_lik(self, xs, zs):
    """Bernoulli log-likelihood, summing over every image n and pixel i
    in image n.

    log p(x | z) = log Bernoulli(x | logits = neural_network(z))
     = sum_{n=1}^N sum_{i=1}^{28*28} log Bernoulli (x_{n,i} | logits_{n,i})
    """
    return tf.reduce_sum(
        bernoulli.logpmf(xs['x'], logits=self.generative_network(zs['z'])))

  def sample_prior(self, n):
    """
    p ~ some complex distribution induced by
    z ~ N(0, 1), p = neural_network(z)
    """
    z = tf.random_normal([n, self.n_vars])
    # Note the output of this is not prior samples, but just the
    # success probability, i.e., the hidden representation learned
    # by the neural network.
    return self.generative_network(z)


def inference_network(x):
  """Inference network to parameterize variational family. It takes
  data as input and outputs the variational parameters.

  mu, sigma = neural_network(x)
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

  mu = tf.reshape(params[:, :d], [-1])
  sigma = tf.reshape(tf.nn.softplus(params[:, d:]), [-1])
  return mu, sigma


ed.set_seed(42)

M = 128  # batch size during training
d = 10  # latent dimension
DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# MODEL
model = NormalBernoulli(d)

# INFERENCE
x_ph = tf.placeholder(tf.float32, [M, 28 * 28])
mu, sigma = inference_network(x_ph)
qz = Normal(mu=mu, sigma=sigma)

# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {'x': x_ph}
inference = ed.ReparameterizationKLKLqp({'z': qz}, data, model)
with tf.variable_scope("model"):
  optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
  inference.initialize(optimizer=optimizer, use_prettytensor=True)

with tf.variable_scope("model", reuse=True):
  p_rep = tf.sigmoid(model.sample_prior(M))

init = tf.global_variables_initializer()
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

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / M
  print("log p(x) >= {:0.3f}".format(avg_loss))

  # Visualize hidden representations.
  imgs = p_rep.eval()
  for m in range(M):
    imsave(os.path.join(IMG_DIR, '%d.png') % m, imgs[m].reshape(28, 28))
