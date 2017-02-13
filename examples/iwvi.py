#!/usr/bin/env python
"""A demo of how to develop new inference algorithms in Edward. Here
we implement importance-weighted variational inference. We test it on
logistic regression.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.inferences import VariationalInference
from edward.models import Bernoulli, Normal, RandomVariable
from edward.util import copy, log_mean_exp
from scipy.special import expit


class IWVI(VariationalInference):
  """Importance-weighted variational inference.

  Uses importance sampling to produce an improved lower bound on the
  log marginal likelihood. It is the core idea behind
  importance-weighted autoencoders (Burda et al. (2016)).
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

    based on the reparameterization trick.
    """
    # Form vector of K log importance weights.
    log_w = []
    for k in range(self.K):
      scope = 'inference_' + str(id(self)) + '/' + str(k)
      z_sample = {}
      q_log_prob = 0.0
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope=scope)
        z_sample[z] = qz_copy
        q_log_prob += tf.reduce_sum(qz_copy.log_prob(qz_copy))

      p_log_prob = 0.0
      for z in six.iterkeys(self.latent_vars):
        # Copy p(z), swapping its conditioning set with samples
        # from variational distribution.
        z_copy = copy(z, z_sample, scope=scope)
        p_log_prob += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          # Copy p(x | z), swapping its conditioning set with samples
          # from variational distribution.
          x_copy = copy(x, z_sample, scope=scope)
          p_log_prob += tf.reduce_sum(x_copy.log_prob(qx))

      log_w += [p_log_prob - q_log_prob]

    loss = -log_mean_exp(log_w)
    grads = tf.gradients(loss, [v._ref() for v in var_list])
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars


ed.set_seed(42)
N = 5000  # number of data points
D = 10  # number of features

# DATA
w_true = np.random.randn(D)
X_data = np.random.randn(N, D)
p = expit(np.dot(X_data, w_true))
y_data = np.array([np.random.binomial(1, i) for i in p])

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
y = Bernoulli(logits=ed.dot(X, w))

# INFERENCE
qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

inference = IWVI({w: qw}, data={X: X_data, y: y_data})
inference.run(K=5, n_iter=1000)

# CRITICISM
print("Mean squared error in true values to inferred posterior mean:")
print(tf.reduce_mean(tf.square(w_true - qw.mean())).eval())
