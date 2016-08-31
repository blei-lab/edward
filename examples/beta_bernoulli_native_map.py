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

from edward.models import Bernoulli, Beta, PointMass

ed.set_seed(42)

p = Beta(a=1.0, b=1.0)
x = Bernoulli(p=tf.ones(10)*p)

data = {x: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

qp_params = tf.nn.sigmoid(tf.Variable(tf.random_normal([1])))
qp = PointMass(params=qp_params)

inference = ed.MAP({p: qp}, data)
inference.run(n_iter=50, n_print=10)
