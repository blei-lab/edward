#!/usr/bin/env python
"""Sparse Gamma deep exponential family (Ranganath et al., 2015). We
apply it as a topic model on the collection of NIPS 2011 conference
papers.

ELBO converges to roughly ~2.3e6 with Gamma q, RBKLqp, and learning
rate of 1e-4 after ~85k iterations / ~20 minutes on a Titan X (Pascal).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Gamma, Poisson, Normal, PointMass, \
    TransformedDistribution

ed.set_seed(42)

# DATA
DATA_DIR = "~/data/nips"
DATA_DIR = os.path.expanduser(DATA_DIR)
x_train = np.loadtxt(os.path.join(DATA_DIR, 'nips_train.csv'))
x_test = np.loadtxt(os.path.join(DATA_DIR, 'nips_test.csv'))

N = x_train.shape[0]  # number of documents
D = x_train.shape[1]  # vocabulary size
shape = 0.1  # gamma shape parameter
K = [100, 40, 15]  # number of components per layer

# MODEL
W2 = Gamma(0.1, 0.3, sample_shape=[K[2], K[1]])
W1 = Gamma(0.1, 0.3, sample_shape=[K[1], K[0]])
W0 = Gamma(0.1, 0.3, sample_shape=[K[0], D])

z3 = Gamma(0.1, 0.1, sample_shape=[N, K[2]])
z2 = Gamma(shape, shape / tf.matmul(z3, W2))
z1 = Gamma(shape, shape / tf.matmul(z2, W1))
x = Poisson(tf.matmul(z1, W0))


# INFERENCE
def pointmass_q(shape):
  min_mean = 1e-3
  mean_init = tf.random_normal(shape)
  rv = PointMass(tf.maximum(tf.nn.softplus(tf.Variable(mean_init)), min_mean))
  return rv


def gamma_q(shape):
  # Parameterize Gamma q's via shape and scale, with softplus unconstraints.
  min_shape = 1e-3
  min_scale = 1e-5
  # min_gamma_sample = 1e-10
  shape_init = 0.5 + 0.1 * tf.random_normal(shape)
  scale_init = 0.1 * tf.random_normal(shape)
  rv = Gamma(tf.maximum(tf.nn.softplus(tf.Variable(shape_init)),
                        min_shape),
             tf.maximum(1.0 / tf.nn.softplus(tf.Variable(scale_init)),
                        1.0 / min_scale))
  return rv


def lognormal_q(shape):
  min_scale = 1e-5
  loc_init = tf.random_normal(shape)
  scale_init = 0.1 + tf.random_normal(shape)
  rv = TransformedDistribution(
      distribution=Normal(
          tf.Variable(loc_init),
          tf.maximum(tf.nn.softplus(tf.Variable(scale_init)), min_scale)),
      bijector=tf.contrib.distributions.bijectors.Exp())
  return rv


qW2 = pointmass_q(W2.shape)
qW1 = pointmass_q(W1.shape)
qW0 = pointmass_q(W0.shape)
qz3 = gamma_q(z3.shape)
qz2 = gamma_q(z2.shape)
qz1 = gamma_q(z1.shape)
# qz3 = lognormal_q(z3.shape)
# qz2 = lognormal_q(z2.shape)
# qz1 = lognormal_q(z1.shape)

# We apply variational EM with E-step over local variables
# and M-step to point estimate the global weight matrices.
# inference_e = ed.KLqp({z1: qz1, z2: qz2, z3: qz3},
#                       data={x: x_train, W0: qW0, W1: qW1, W2: qW2})
inference_e = ed.ScoreRBKLqp({z1: qz1, z2: qz2, z3: qz3},
                             data={x: x_train, W0: qW0, W1: qW1, W2: qW2})
inference_m = ed.MAP({W0: qW0, W1: qW1, W2: qW2},
                     data={x: x_train, z1: qz1, z2: qz2, z3: qz3})
optimizer_m = tf.train.RMSPropOptimizer(1e-4)
optimizer_e = tf.train.RMSPropOptimizer(1e-4)
# inference_e.initialize(optimizer=optimizer_e,
#                        n_iter=int(1e6),
#                        n_print=100,
#                        logdir='~/log/def')
inference_e.initialize(optimizer=optimizer_e,
                       n_iter=int(1e6),
                       n_print=100,
                       n_samples=50,
                       logdir='~/log/def')
inference_m.initialize(optimizer=optimizer_m)

# # to compute held-out perplexity during training
# N_test = x_test.shape[1]
# perplexity = tf.exp(tf.reduce_sum() / N_test)

# sess = ed.get_session()
tf.global_variables_initializer().run()

for _ in range(inference_e.n_iter):
  info_dict_e = inference_e.update()
  info_dict_m = inference_m.update()
  inference_e.print_progress(info_dict_e)

  # Training perplexity.
  # avg_loss = avg_loss / n_iter_per_epoch
  # avg_loss = avg_loss / M
  # print("log p(x) >= {:0.3f}".format(avg_loss))

  # Held-out perplexity.
  # TODO
