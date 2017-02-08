#!/usr/bin/env python
"""Generative adversarial network for MNIST (Goodfellow et al., 2014).
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
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data


def generative_network(z):
  h1 = slim.fully_connected(z, 128, activation_fn=tf.nn.relu)
  x = slim.fully_connected(h1, 784, activation_fn=tf.sigmoid)
  return x


def discriminative_network(x):
  """Outputs probability in logits."""
  h1 = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
  logit = slim.fully_connected(h1, 1, activation_fn=None)
  return logit


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

M = 128  # batch size during training
d = 100  # latent dimension

DATA_DIR = "data/mnist"
IMG_DIR = "img"

if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)

# DATA. MNIST batches are fed at training time.
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
x_ph = tf.placeholder(tf.float32, [M, 784])

# MODEL
with tf.variable_scope("Gen"):
  z = Uniform(a=tf.zeros([M, d]) - 1.0, b=tf.ones([M, d]))
  x = generative_network(z)

# INFERENCE
optimizer = tf.train.AdamOptimizer()
optimizer_d = tf.train.AdamOptimizer()

inference = ed.GANInference(
    data={x: x_ph}, discriminator=discriminative_network)
inference.initialize(
    optimizer=optimizer, optimizer_d=optimizer,
    n_iter=15000, n_print=1000)

sess = ed.get_session()
tf.global_variables_initializer().run()

idx = np.random.randint(M, size=16)
i = 0
for t in range(inference.n_iter):
  if t % inference.n_print == 0:
    samples = sess.run(x)
    samples = samples[idx, ]

    fig = plot(samples)
    plt.savefig(os.path.join(IMG_DIR, '{}.png').format(
        str(i).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    i += 1

  x_batch, _ = mnist.train.next_batch(M)
  info_dict = inference.update(feed_dict={x_ph: x_batch})
  inference.print_progress(info_dict)
