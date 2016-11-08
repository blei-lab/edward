#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.

The model is written in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Beta
from edward.stats import bernoulli, beta


class BetaBernoulli:
  """p(x, p) = Bernoulli(x | p) * Beta(p | 1, 1)"""
  def log_prob(self, xs, zs):
    log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
    log_lik = tf.reduce_sum(bernoulli.logpmf(xs['x'], p=zs['p']))
    return log_lik + log_prior

  def sample_prior(self):
    """p ~ p(p)"""
    return {'p': beta.sample(a=1.0, b=1.0)}

  def sample_likelihood(self, zs):
    """x | p ~ p(x | p)"""
    return {'x': bernoulli.sample(p=tf.ones(10) * zs['p'])}


def T(xs, zs):
  return tf.reduce_mean(tf.cast(xs['x'], tf.float32))


ed.set_seed(42)
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

model = BetaBernoulli()

qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(a=qp_a, b=qp_b)

inference = ed.KLqp({'p': qp}, data, model)
inference.run(n_iter=200)

print(ed.ppc(T, data, model_wrapper=model))
