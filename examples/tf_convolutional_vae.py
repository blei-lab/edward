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
import os
import prettytensor as pt
import tensorflow as tf

from tf_convolutional_vae_util import deconv2d
from edward.models import Normal
from edward.stats import bernoulli
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

N_MINIBATCH = 128
DATA_DIR = "data/mnist"
IMG_DIR = "img"


class NormalBernoulli:
  """
  Each binarized pixel in an image is modeled by a Bernoulli
  likelihood. The success probability for each pixel is the output
  of a neural network that takes samples from a normal prior as
  input.

  p(x, z) = Bernoulli(x | p = neural_network(z)) Normal(z; 0, I)
  """
  def __init__(self, n_vars):
    self.n_vars = n_vars  # number of local latent variables

  def neural_network(self, z):
    """p = neural_network(z)"""
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
      return (pt.wrap(z).
              reshape([N_MINIBATCH, 1, 1, self.n_vars]).
              deconv2d(3, 128, edges='VALID').
              deconv2d(5, 64, edges='VALID').
              deconv2d(5, 32, stride=2).
              deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
              flatten()).tensor

  def log_lik(self, xs, zs):
    """
    Bernoulli log-likelihood, summing over every image n and pixel i
    in image n.

    log p(x | z) = log Bernoulli(x | p = neural_network(z))
     = sum_{n=1}^N sum_{i=1}^{28*28} log Bernoulli (x_{n,i} | p_{n,i})
    """
    return tf.reduce_sum(
        bernoulli.logpmf(xs['x'], p=self.neural_network(zs['z'])))

  def sample_prior(self, n):
    """
    p ~ some complex distribution induced by
    z ~ N(0, 1), p = neural_network(z)
    """
    z = tf.random_normal([n, self.n_vars])
    # Note the output of this is not prior samples, but just the
    # success probability, i.e., the hidden representation learned
    # by the neural network.
    return self.neural_network(z)


def neural_network(x):
  """
  Inference network to parameterize variational family. It takes
  data as input and outputs the variational parameters.

  mu, sigma = neural_network(x)
  """
  n_vars = 10
  with pt.defaults_scope(activation_fn=tf.nn.elu,
                         batch_normalize=True,
                         learned_moments_update_rate=0.0003,
                         variance_epsilon=0.001,
                         scale_after_normalization=True):
    params = (pt.wrap(x).
              reshape([N_MINIBATCH, 28, 28, 1]).
              conv2d(5, 32, stride=2).
              conv2d(5, 64, stride=2).
              conv2d(5, 128, edges='VALID').
              dropout(0.9).
              flatten().
              fully_connected(n_vars * 2, activation_fn=None)).tensor

  # Return list of vectors where mean[i], stddev[i] are the
  # parameters of the local variational factor for data point i.
  mu = tf.reshape(params[:, :n_vars], [-1])
  sigma = tf.reshape(tf.sqrt(tf.exp(params[:, n_vars:])), [-1])
  return [mu, sigma]


ed.set_seed(42)
model = NormalBernoulli(n_vars=10)

# Use the variational model
# q(z | x) = prod_{n=1}^n Normal(z_n | mu, sigma = neural_network(x_n))
# It is a distribution of the latent variables z_n for each data
# point x_n. We use neural_network() to globally parameterize the local
# variational factors q(z_n | x).
# We also do data subsampling during inference. Therefore we only need
# to explicitly represent the variational factors for a mini-batch,
# q(z_{batch} | x) = prod_{m=1}^{n_data}
#                    Normal(z_m | mu, sigma = neural_network(x_m))
x_ph = ed.placeholder(tf.float32, [N_MINIBATCH, 28 * 28])
mu, sigma = neural_network(x_ph)
qz = Normal(mu=mu, sigma=sigma)

# MNIST batches are fed at training time.
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
# Bind p(x, z) and q(z | x) to the same TensorFlow placeholder for x.
data = {'x': x_ph}

sess = ed.get_session()
inference = ed.MFVI({'z': qz}, data, model)
with tf.variable_scope("model") as scope:
  optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
  inference.initialize(optimizer=optimizer, use_prettytensor=True)
with tf.variable_scope("model", reuse=True) as scope:
  p_rep = model.sample_prior(N_MINIBATCH)

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
    x_train, _ = mnist.train.next_batch(N_MINIBATCH)
    info_dict = inference.update(feed_dict={x_ph: x_train})
    avg_loss += info_dict['loss']

  # Take average over all ELBOs during the epoch, and over minibatch
  # of data points (images).
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / N_MINIBATCH

  # Print a lower bound to the average marginal likelihood for an
  # image.
  print("log p(x) >= {:0.3f}".format(avg_loss))

  imgs = p_rep.eval()
  for b in range(N_MINIBATCH):
    if not os.path.exists(IMG_DIR):
      os.makedirs(IMG_DIR)

    imsave(os.path.join(IMG_DIR, '%d.png') % b,
           imgs[b].reshape(28, 28))
