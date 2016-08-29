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


class BernoulliPosterior:
  """p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)"""
  def log_prob(self, xs, zs):
    return bernoulli.logpmf(zs['p'], 0.6)


ed.set_seed(42)
model = BernoulliPosterior()

qp_p = tf.nn.sigmoid(tf.Variable(tf.random_normal([1])))
qp = Bernoulli(p=qp_p)

inference = ed.MFVI({'z': qz}, model_wrapper=model)
inference.run(n_samples=int(1e5))
