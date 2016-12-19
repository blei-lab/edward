#!/usr/bin/env python
"""A simple demonstration of how to implement new inference algorithms
in Edward. Here we implement importance-weighted variational
inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.inferences import KLqp
from edward.models import Beta
from edward.stats import bernoulli, beta
from edward.util import copy, log_mean_exp


class IWVI(KLqp):
  """
  Importance-weighted variational inference. Uses importance
  sampling to produce an improved lower bound on the log marginal
  likelihood.

  It is the core idea behind importance-weighted autoencoders (Burda
  et al. (2016)). IWAEs are the special case when the probabilistic
  model is among a specific class of deep generative models, and the
  variational model is parameterized with an inference network.

  Notes
  -----
  `IWVI` is implemented by inheriting from Kullback-Leibler (black box)
  variational inference (`KLqp`). The loss function to optimize is
  modified to include importance weights. It is only implemented to
  work on model wrappers and not Edward's native modeling language.
  """
  def __init__(self, *args, **kwargs):
    super(IWVI, self).__init__(*args, **kwargs)

  def initialize(self, K=5, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    K : int, optional
      Number of importance samples.
    """
    self.K = K
    return super(IWVI, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -E_{q(z^1; \lambda), ..., q(z^K; \lambda)} [
      \log 1/K \sum_{k=1}^K p(x, z^k)/q(z^k; \lambda) ]

    based on the score function estimator. (Paisley et al., 2012)

    Computed by sampling from :math:`q(z;\lambda)` and evaluating
    the expectation using Monte Carlo sampling. Note there is a
    difference between the number of samples to approximate the
    expectations (`n_samples`) and the number of importance
    samples to determine how many expectations (`K`).
    """
    x = self.data
    # Form n_samples x K matrix of log importance weights.
    log_w = []
    for s in range(self.n_samples * self.K):
      z_sample = {}
      q_log_prob = 0.0
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob += tf.reduce_sum(qz.log_prob(tf.stop_gradient(z_sample[z])))

      p_log_prob = self.model_wrapper.log_prob(x, z_sample)
      log_w += [p_log_prob - q_log_prob]

    log_w = tf.reshape(log_w, [self.n_samples, self.K])
    # Take log mean exp across importance weights (columns).
    losses = log_mean_exp(log_w, 1)
    loss = -tf.reduce_mean(losses)

    if var_list is None:
      var_list = tf.trainable_variables()

    grads = tf.gradients(
        -tf.reduce_mean(q_log_prob * tf.stop_gradient(losses)),
        [v.ref() for v in var_list])
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars


class BetaBernoulli:
  """p(x, p) = Bernoulli(x | z) * Beta(p | 1, 1)"""
  def log_prob(self, xs, zs):
    log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
    log_lik = tf.reduce_sum(bernoulli.logpmf(xs['x'], p=zs['p']))
    return log_lik + log_prior


ed.set_seed(42)
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

model = BetaBernoulli()

qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(a=qp_a, b=qp_b)

inference = IWVI({'p': qp}, data, model)
inference.run(K=5, n_iter=500)
