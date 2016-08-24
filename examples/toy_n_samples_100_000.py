#!/usr/bin/env python
"""
This is just to show how fast we can do a gradient descent
with 100,000 samples (!).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Bernoulli
from edward.stats import bernoulli


class BernoulliModel:
  """p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)"""
  def __init__(self, p):
    self.p = p

  def log_prob(self, xs, zs):
    return bernoulli.logpmf(zs['z'], p)


ed.set_seed(42)
p = tf.constant(0.6)
model = BernoulliModel(p)
qz = Bernoulli()

inference = ed.MFVI({'z': qz}, model_wrapper=model)
inference.run(n_samples=int(1e5))
