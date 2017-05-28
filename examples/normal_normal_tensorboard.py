#!/usr/bin/env python
"""Normal-normal model using variational inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

ed.set_seed(42)

# DATA
x_data = np.array([0.0] * 50)

# MODEL: Normal-Normal with known variance
mu = Normal(loc=0.0, scale=1.0, name='mu')
x = Normal(loc=tf.ones(50) * mu, scale=1.0, name='x')

# INFERENCE
qmu_mu = tf.Variable(tf.random_normal([]), name='qmu_mu')
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])), name='qmu_sigma')
qmu = Normal(loc=qmu_mu, scale=qmu_sigma, name='qmu')

# analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
inference = ed.KLqp({mu: qmu}, data={x: x_data})
inference.run(logdir='log')
