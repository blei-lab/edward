#!/usr/bin/env python
"""Convolutional variational auto-encoder for binarized MNIST.

The neural networks are written with TensorFlow Slim.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  with slim.arg_scope([slim.conv2d_transpose],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(z, [M, 1, 1, d])
    net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
    net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
    net = slim.conv2d_transpose(net, 32, 5, stride=2)
    net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
    net = slim.flatten(net)
    return net


def inference_network(x):
  """Inference network to parameterize variational model. It takes
  data as input and outputs the variational parameters.

  mu, sigma = neural_network(x)
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.elu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True}):
    net = tf.reshape(x, [M, 28, 28, 1])
    net = slim.conv2d(net, 32, 5, stride=2)
    net = slim.conv2d(net, 64, 5, stride=2)
    net = slim.conv2d(net, 128, 5, padding='VALID')
    net = slim.dropout(net, 0.9)
    net = slim.flatten(net)
    params = slim.fully_connected(net, d * 2, activation_fn=None)

  mu = params[:, :d]
  sigma = tf.nn.softplus(params[:, d:])
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
z = Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
logits = generative_network(z.value())
x = Bernoulli(logits=logits)

# INFERENCE
x_ph = tf.placeholder(tf.float32, [M, 28 * 28])
mu, sigma = inference_network(x_ph)
qz = Normal(mu=mu, sigma=sigma)

# Bind p(x, z) and q(z | x) to the same placeholder for x.
data = {x: x_ph}
inference = ed.KLqp({z: qz}, data)
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)

hidden_rep = tf.sigmoid(logits)

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
  imgs = hidden_rep.eval()
  for m in range(M):
    imsave(os.path.join(IMG_DIR, '%d.png') % m, imgs[m].reshape(28, 28))
