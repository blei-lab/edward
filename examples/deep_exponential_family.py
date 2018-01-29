# -*- coding: utf-8 -*-
"""Sparse Gamma deep exponential family (Ranganath et al., 2015). We
apply it as a topic model on the collection of NIPS 2011 conference
papers.

The loss function can sometimes erroneously output a negative value or
NaN. This happens when the samples from the variational approximation
are numerically zero, which causes Gamma log probs to output inf.

With default settings (in particular, with log normal variational
approximation), it takes ~62s per epoch on a Titan X (Pascal).
Following results are on epoch 12.

10000/10000 [100%] ██████████████████████████████ Elapsed: 62s
Negative log-likelihood <= -1060649.607
Perplexity <= 0.205
Topic 0: let distribution set strategy distributions given learning
    information use property
Topic 1: functions problem risk function submodular cut level
    clustering sets performance
Topic 2: action value learning regret reward actions algorithm optimal
    state return
Topic 3: posterior stochastic approach information based using prior
    mean divergence since
Topic 4: player inference game propagation experts static query expert
    base variables
Topic 5: algorithm set loss weak algorithms optimal submodular online
    cost setting
Topic 6: sparse sparsity norm solution learning penalty greedy
    structure wise regularization
Topic 7: learning training linear kernel using coding accuracy
    performance dataset based
Topic 8: object categories image features examples classes images
    class objects visual
Topic 9: data manifold matrix points dimensional point low linear
    gradient optimization

A Gamma variational approximation produces worse results, which is
likely due to the high variance in stochastic gradients. It takes ~2
minutes per epoch on a Titan X (Pascal). Following results are on
epoch 12.

Negative log-likelihood <= 3738025.615
Perplexity <= 266.623
Topic 0: reasons posterior tion using similar tools university input
    computed refers
Topic 1: expected since much related rate defined optimization vector
    thus neurons
Topic 2: large linear given table shown true drop classification
    constraints current
Topic 3: proposed processing estimated better values gaussian form
    test true setting
Topic 4: see methods local several rate processing general vector
    enables section
Topic 5: thus case methods image dataset models different instead new
    respectively
Topic 6: based consider samples step object see kernel since problem
    training
Topic 7: approaches linear computing show gaussian data expected
    analysis well proof
Topic 8: fig point kernel bayesian solution applications results
    follows regression computer
Topic 9: conference optimization training pages maximum learning
    dataset performance state inference
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from datetime import datetime
from edward.models import Gamma, Poisson, Normal, PointMass, \
    TransformedDistribution
from edward.util import Progbar
from observations import nips

tf.flags.DEFINE_string("data_dir", default="~/data", help="")
tf.flags.DEFINE_string("logdir", default="~/log/def/", help="")
tf.flags.DEFINE_list("K", default=[100, 30, 15],
                     help="Number of components per layer.")
tf.flags.DEFINE_string("q", default="lognormal",
                       help="Choice of q; 'lognormal' or 'gamma'.")
tf.flags.DEFINE_float("shape", default=0.1, help="Gamma shape parameter.")
tf.flags.DEFINE_float("lr", default=1e-4, help="Learning rate step-size.")

FLAGS = tf.flags.FLAGS
FLAGS.data_dir = os.path.expanduser(FLAGS.data_dir)
FLAGS.logdir = os.path.expanduser(FLAGS.logdir)
timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
FLAGS.logdir += timestamp + '_' + '_'.join([str(ks) for ks in FLAGS.K]) + \
    '_q_' + str(FLAGS.q) + '_lr_' + str(FLAGS.lr)


def pointmass_q(shape, name=None):
  with tf.variable_scope(name, default_name="pointmass_q"):
    min_mean = 1e-3
    mean = tf.get_variable("mean", shape)
    rv = PointMass(tf.maximum(tf.nn.softplus(mean), min_mean))
    return rv


def gamma_q(shape, name=None):
  # Parameterize Gamma q's via shape and scale, with softplus unconstraints.
  with tf.variable_scope(name, default_name="gamma_q"):
    min_shape = 1e-3
    min_scale = 1e-5
    shape = tf.get_variable(
        "shape", shape,
        initializer=tf.random_normal_initializer(mean=0.5, stddev=0.1))
    scale = tf.get_variable(
        "scale", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    rv = Gamma(tf.maximum(tf.nn.softplus(shape), min_shape),
               tf.maximum(1.0 / tf.nn.softplus(scale), 1.0 / min_scale))
    return rv


def lognormal_q(shape, name=None):
  with tf.variable_scope(name, default_name="lognormal_q"):
    min_scale = 1e-5
    loc = tf.get_variable("loc", shape)
    scale = tf.get_variable(
        "scale", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    rv = TransformedDistribution(
        distribution=Normal(loc, tf.maximum(tf.nn.softplus(scale), min_scale)),
        bijector=tf.contrib.distributions.bijectors.Exp())
    return rv


def main(_):
  ed.set_seed(42)

  # DATA
  x_train, metadata = nips(FLAGS.data_dir)
  documents = metadata['columns']
  words = metadata['rows']

  # Subset to documents in 2011 and words appearing in at least two
  # documents and have a total word count of at least 10.
  doc_idx = [i for i, document in enumerate(documents)
             if document.startswith('2011')]
  documents = [documents[doc] for doc in doc_idx]
  x_train = x_train[:, doc_idx]
  word_idx = np.logical_and(np.sum(x_train != 0, 1) >= 2,
                            np.sum(x_train, 1) >= 10)
  words = [word for word, idx in zip(words, word_idx) if idx]
  x_train = x_train[word_idx, :]
  x_train = x_train.T

  N = x_train.shape[0]  # number of documents
  D = x_train.shape[1]  # vocabulary size

  # MODEL
  W2 = Gamma(0.1, 0.3, sample_shape=[FLAGS.K[2], FLAGS.K[1]])
  W1 = Gamma(0.1, 0.3, sample_shape=[FLAGS.K[1], FLAGS.K[0]])
  W0 = Gamma(0.1, 0.3, sample_shape=[FLAGS.K[0], D])

  z3 = Gamma(0.1, 0.1, sample_shape=[N, FLAGS.K[2]])
  z2 = Gamma(FLAGS.shape, FLAGS.shape / tf.matmul(z3, W2))
  z1 = Gamma(FLAGS.shape, FLAGS.shape / tf.matmul(z2, W1))
  x = Poisson(tf.matmul(z1, W0))

  # INFERENCE
  qW2 = pointmass_q(W2.shape)
  qW1 = pointmass_q(W1.shape)
  qW0 = pointmass_q(W0.shape)
  if FLAGS.q == 'gamma':
    qz3 = gamma_q(z3.shape)
    qz2 = gamma_q(z2.shape)
    qz1 = gamma_q(z1.shape)
  else:
    qz3 = lognormal_q(z3.shape)
    qz2 = lognormal_q(z2.shape)
    qz1 = lognormal_q(z1.shape)

  # We apply variational EM with E-step over local variables
  # and M-step to point estimate the global weight matrices.
  inference_e = ed.KLqp({z1: qz1, z2: qz2, z3: qz3},
                        data={x: x_train, W0: qW0, W1: qW1, W2: qW2})
  inference_m = ed.MAP({W0: qW0, W1: qW1, W2: qW2},
                       data={x: x_train, z1: qz1, z2: qz2, z3: qz3})

  optimizer_e = tf.train.RMSPropOptimizer(FLAGS.lr)
  optimizer_m = tf.train.RMSPropOptimizer(FLAGS.lr)
  kwargs = {'optimizer': optimizer_e,
            'n_print': 100,
            'logdir': FLAGS.logdir,
            'log_timestamp': False}
  if FLAGS.q == 'gamma':
    kwargs['n_samples'] = 30
  inference_e.initialize(**kwargs)
  inference_m.initialize(optimizer=optimizer_m)

  sess = ed.get_session()
  tf.global_variables_initializer().run()

  n_epoch = 20
  n_iter_per_epoch = 10000
  for epoch in range(n_epoch):
    print("Epoch {}".format(epoch))
    nll = 0.0

    pbar = Progbar(n_iter_per_epoch)
    for t in range(1, n_iter_per_epoch + 1):
      pbar.update(t)
      info_dict_e = inference_e.update()
      info_dict_m = inference_m.update()
      nll += info_dict_e['loss']

    # Compute perplexity averaged over a number of training iterations.
    # The model's negative log-likelihood of data is upper bounded by
    # the variational objective.
    nll /= n_iter_per_epoch
    perplexity = np.exp(nll / np.sum(x_train))
    print("Negative log-likelihood <= {:0.3f}".format(nll))
    print("Perplexity <= {:0.3f}".format(perplexity))

    # Print top 10 words for first 10 topics.
    qW0_vals = sess.run(qW0)
    for k in range(10):
      top_words_idx = qW0_vals[k, :].argsort()[-10:][::-1]
      top_words = " ".join([words[i] for i in top_words_idx])
      print("Topic {}: {}".format(k, top_words))

if __name__ == "__main__":
  tf.app.run()
