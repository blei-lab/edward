#!/usr/bin/env python
"""Bernoulli posterior.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, PointMass
from edward.stats import bernoulli


class BernoulliPosterior:
  """p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)"""
  def log_prob(self, xs, zs):
    return bernoulli.logpmf(zs['p'], p=0.6)


ed.set_seed(42)
model = BernoulliPosterior()

qp_p = tf.sigmoid(tf.Variable(tf.random_normal([])))
qp = Bernoulli(p=qp_p)

inference = ed.KLqp({'p': qp}, model_wrapper=model)
inference.run(n_iter=100, n_samples=5)
