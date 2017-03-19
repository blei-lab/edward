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
mu = Normal(mu=0.0, sigma=1.0)
x = Normal(mu=tf.ones(50) * mu, sigma=1.0)

# INFERENCE
qmu_mu = tf.Variable(tf.random_normal([]))
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)

# analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
inference = ed.KLqp({mu: qmu}, data={x: x_data})
inference.run()
