#!/usr/bin/env python
"""Generative adversarial network for toy mixture data.
(Goodfellow et al., 2014)

See also noise contrastive estimation (Gutmann and Hyvarinen, 2010).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal
from tensorflow.contrib import slim

plt.style.use('ggplot')


def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [1, -1]
  stds = [0.1, 0.1]
  x = np.zeros((N, 1), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.normal(mus[k], stds[k])

  return x


def generative_network(z):
  """Generative network which takes latent variables as input and
  outputs data.
  """
  net = z
  net = slim.fully_connected(net, 25)
  net = slim.fully_connected(net, N, activation_fn=None)
  net = slim.flatten(net)
  return net


def discriminative_network(x):
  """Discriminative network which takes real or fake data as input
  and outputs the probability that the data is real (in logit
  parameterization).
  """
  net = x
  net = slim.fully_connected(net, 25)
  net = slim.fully_connected(net, N, activation_fn=None)
  net = slim.flatten(net)
  return net


ed.set_seed(42)

N = 100  # data set size
d = 2  # latent variable dimension

# DATA
x_data = build_toy_dataset(N)
plt.title("Simulated dataset")
plt.hist(x_data, bins=np.arange(-2, 2 + 0.5, 0.1))
plt.show()

# MODEL. It is a neural network applied to random noise.
z = Normal(mu=tf.zeros([N, d]), sigma=tf.ones([N, d]))
x = generative_network(z)

# INFERENCE. The model is augmented into a supervised setup with fixed labels,
# { (x'_1, 0), ..., (x'_n, 0), (x_1, 1), ..., (x_N, 1) },
# where x'_1, ..., x'_n are generated from the model.
# We perform MAP on this augmented space. It maximizes
# log p(yreal | D(x_data)) + E_{p(x)} [ log p(yfake | D(x)) ]
y_fake = Bernoulli(logits=discriminative_network(x))
y_real = Bernoulli(logits=discriminative_network(x_data))

data = {y_real: tf.ones([N]), y_fake: tf.zeros([N])}
inference = ed.MAP({}, data)
inference.initialize()

init = tf.initialize_all_variables()
init.run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

# CRITICISM. Visualize fit of generative model.
sess = ed.get_session()
x_fake = sess.run(x)
plt.title("Generated dataset")
# plt.hist(x_fake, bins=np.arange(-2, 2 + 0.5, 0.1))
plt.hist(x_fake)
plt.show()
