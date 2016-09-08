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
import tensorflow as tf

from edward.models import Bernoulli, Normal, PointMass
from word_embeddings_util import \
    maybe_download, read_data, build_dataset, generate_batch

# TODO
# Challenges:
# + Undirected/conditionally specified-ness. Require ``tie``.
# + Data subsampling with undirected models and the lagging in the
# previous data subsample.
# + Sparse matrix. Require sparse sample support for discrete random
# variables.

ed.set_seed(42)

## DATA

# Data is a N x V matrix, indexed by i = (n, v), where n is the
# position in the text and v indexes the vocabulary.

# Download the data and read it into a list of strings.
url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)

N = len(words)  # total number of words
V = 50000  # size of the vocabulary

print('Data size:', N)
print('Vocabulary size:', V)

# Build the dictionary and replace rare words with UNK token.
data, count, dictionary, reverse_dictionary = build_dataset(words, V)
del words  # remove in order to reduce memory
print('Most common words:', count[:5])
print('Sample data:', data[:10], [reverse_dictionary[i] for i in data[:10]])

## MODEL

# We will do batch training.
# Define a subgraph of the graphical model, that is, a word
# embedding model of M data points rather than N data points.

M = 128  # batch size number of words
K = 128  # dimensionality of embedding


def context(i):
  """Define context of a data point i as {j != i | i - w <= j <= i + w},
  for fixed w.
  Here, context returns just the row indices.
  """
  w = 1
  return [j for j in range(i - w, i + w + 1)
          if j != i and j >= 0 and j <= N]


# Word embeddings form an undirected graphical model. Represent it as a
# directed graphical model, where the top nodes are tied to the bottom
# nodes during inference.
x_tied = tf.Variable(tf.zeros([M], dtype=tf.int32), trainable=False)

# Place prior over embedding (rho) and context (alpha) vectors of
# size K, one for each term in the vocabulary.
rho = Normal(mu=tf.zeros([V, K]), sigma=tf.ones([V, K]))
alpha = Normal(mu=tf.zeros([V, K]), sigma=tf.ones([V, K]))

# Form natural parameters eta[m, v] for each position in the text m
# and vocabulary term v. We vectorize this operation over its columns
# and loop over the rows.
eta = [0] * M
for m in range(M):
  # Get data point i's context.
  # First gather the rows. Then convert from one-hot to labels:
  # (2 * w - 1, V) matrix of {0, 1} values ->
  # (2 * w - 1, ) vector of {0, ..., V - 1} values.
  x_js = tf.gather(x_tied, context(m))
  x_js = tf.argmax(x_js, 1) # TODO must be a more efficient method
  # Sum over context vectors in data point i's context.
  alpha_sum = tf.reduce_sum(tf.gather(alpha, x_js), 0)
  # Take linear combination of data point i's embedding vector with
  # context vectors in its context.
  eta[m] = ed.dot(rho, alpha_sum)

eta = tf.pack(eta)

# Posit M x V data model according to the M x V natural parameters.
x = Bernoulli(logits=eta)

## INFERENCE

# Summarize the posterior of rho and alpha using points.
qrho = PointMass(params=tf.Variable(tf.random_uniform([V, K], -1.0, 1.0)))
qalpha = PointMass(params=tf.Variable(tf.random_uniform([V, K], -1.0, 1.0)))

# MAP maximizes the sum of the conditional likelihoods with log-prior
# penalties. A biased SGD with this objective is equivalent to
# negative sampling with the CBOW model.
x_ph = tf.placeholder(tf.int32, shape=[M, V])
inference = ed.MAP({rho: qrho, alpha: qalpha},
                   data={x: x_ph}, tie={x_tied: x})

inference.initialize()
sess = ed.get_session()

# TODO change generate_batch() to generate the right data structure.
# TODO use a M-vector and implicitly represent it as a M x V matrix
# Still, a M x V matrix is a little crazy.. better to store it as a
# sparse M x V matrix. this requires support for sparse random
# variables
data_idx = 0  # running global index for generate_batch
for t in range(1000):
  x_data = generate_batch(batch_size=M)
  _, loss = sess.run([inference.train, inference.loss],
                     feed_dict={x_ph: x_data})
  inference.print_progress(t, loss)

## CRITICISM

# TODO TSNE
