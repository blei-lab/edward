#!/usr/bin/env python
"""
A probabilistic model for word embeddings (Bengio et al., 2003; Mikolov
et al., 2013).

We follow the notation used in exponential family embeddings (Rudolph
et al., 2016). It defines a Bernoulli embedding for matrix data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, PointMass

ed.set_seed(42)

## DATA

# Data is a N x V matrix, indexed by i = (n, v), where n is the
# position in the text and v indexes the vocabulary.
# TODO load in from word embedding examples
#"http://mattmahoney.net/dc/text8.zip"
#https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
#https://www.tensorflow.org/versions/r0.10/tutorials/word2vec/index.html
#x_data = load()
x_data = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]])

N = x_data.shape[0]  # number of words
V = x_data.shape[1]  # size of the vocabulary
K = 2  # dimensionality of embedding

## MODEL


def context(i):
  """Define context of a data point i as {j != i | i - w <= j <= i + w},
  for fixed w.
  Here, context returns just the row indices.
  """
  w = 1
  return [j for j in range(i - w, i + w + 1)
          if j != i and j >= 0 and j <= V]


# Word embeddings form an undirected graphical model. Represent it as a
# directed graphical model, where the top nodes are tied to the bottom
# nodes during inference.
x_ph = tf.zeros([N, V], dtype=tf.int32)

# Place prior over embedding (rho) and context (alpha) vectors of
# size K, one for each term in the vocabulary.
rho = Normal(mu=tf.zeros([V, K]), sigma=tf.ones([V, K]))
alpha = Normal(mu=tf.zeros([V, K]), sigma=tf.ones([V, K]))

# Form natural parameters eta[n, v] for each position in the text n
# and vocabulary term v. We vectorize this operation over its columns
# and loop over the rows.
eta = [0] * N
for n in range(N):
  # Get data point i's context.
  # First gather the rows. Then convert from one-hot to labels:
  # (2 * w - 1, V) matrix of {0, 1} values ->
  # (2 * w - 1, ) vector of {0, ..., V - 1} values.
  x_js = tf.gather(x_ph, context(n))
  x_js = tf.argmax(x_js, 1) # TODO must be a more efficient method
  # Sum over context vectors in data point i's context.
  alpha_sum = tf.reduce_sum(tf.gather(alpha, x_js), 0)
  # Take linear combination of data point i's embedding vector with
  # context vectors in its context.
  eta[n] = ed.dot(rho, alpha_sum)

eta = tf.pack(eta)

# Generate data points according to the natural parameters.
x = Bernoulli(logits=eta)

## INFERENCE

# Summarize the posterior of rho and alpha using points.
qrho = PointMass(params=tf.Variable(tf.random_normal([V, K])))
qalpha = PointMass(params=tf.Variable(tf.random_normal([V, K])))

# MAP maximizes the sum of the conditional likelihoods with log-prior
# penalties. A biased SGD with this objective is equivalent to
# negative sampling with the skip-gram model.
inference = ed.MAP({rho: qrho, alpha: qalpha},
                   data={x: x_data}, tie={x_ph: x})

inference.initialize()
for t in range(1000):
  loss = inference.update()
  inference.print_progress(t, loss)

## CRITICISM

# TODO TSNE
