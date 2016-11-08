#!/usr/bin/env python
"""Normal posterior.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm


class NormalPosterior:
  """p(x, z) = p(z) = p(z | x) = Normal(z; mu, sigma)"""
  def log_prob(self, xs, zs):
    return norm.logpdf(zs['z'], 1.0, 1.0)


ed.set_seed(42)
model = NormalPosterior()

qz_mu = tf.Variable(tf.random_normal([]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

inference = ed.KLqp({'z': qz}, model_wrapper=model)
inference.run(n_iter=250)
