#!/usr/bin/env python
"""Logistic factor analysis on MNIST. Using Monte Carlo EM, with HMC
for the E-step and MAP for the M-step. We fit to just one data
point in MNIST.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import os
import tensorflow as tf

from edward.models import Bernoulli, Empirical, Normal
from scipy.misc import imsave
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data


def generative_network(z):
  """Generative network to parameterize generative model. It takes
  latent variables as input and outputs the likelihood parameters.

  logits = neural_network(z)
  """
  net = slim.fully_connected(z, 28 * 28, activation_fn=None)
  net = slim.flatten(net)
  return net


ed.set_seed(42)

N = 1  # number of data points
d = 10  # latent dimension
DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

# DATA
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
x_train, _ = mnist.train.next_batch(N)

# MODEL
z = Normal(loc=tf.zeros([N, d]), scale=tf.ones([N, d]))
logits = generative_network(z)
x = Bernoulli(logits=logits)

# INFERENCE
n_iter_per_epoch = 100
n_epoch = 1000

T = n_iter_per_epoch * n_epoch
qz = Empirical(params=tf.Variable(tf.random_normal([T, N, d])))

inference_e = ed.HMC({z: qz}, data={x: x_train})
inference_e.initialize(n_print=n_iter_per_epoch)

inference_m = ed.MAP(data={x: x_train, z: qz.params[inference_e.t]})
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference_m.initialize(optimizer=optimizer)

tf.global_variables_initializer().run()

for _ in range(n_epoch):
  avg_loss = 0.0
  for _ in range(n_iter_per_epoch):
    info_dict_e = inference_e.update()
    info_dict_m = inference_m.update()
    avg_loss += info_dict_m['loss']

  inference_e.print_progress(info_dict_e)

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss = avg_loss / n_iter_per_epoch
  avg_loss = avg_loss / N
  print("\nlog p(x) >= {:0.3f}".format(avg_loss))

  # Prior predictive check.
  imgs = x.eval()
  for m in range(N):
    imsave(os.path.join(IMG_DIR, '%d.png') % m, imgs[m].reshape(28, 28))
