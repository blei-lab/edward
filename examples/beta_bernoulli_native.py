#!/usr/bin/env python
"""
A simple coin flipping example. Inspired by Stan's toy example.

Probability model
  Prior: Beta
  Likelihood: Bernoulli
Variational model
  Likelihood: Mean-field Beta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta

ed.set_seed(42)

p = Beta(a=1.0, b=1.0)
x = Bernoulli(p=tf.ones(10) * p)

data = {x: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(a=qp_a, b=qp_b)

inference = ed.MFVI({p: qp}, data)
inference.run(n_iter=10000)
