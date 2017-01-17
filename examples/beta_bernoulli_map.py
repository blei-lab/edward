#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, PointMass

ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
p = Beta(a=1.0, b=1.0)
x = Bernoulli(p=tf.ones(10) * p)

# INFERENCE
qp_params = tf.sigmoid(tf.Variable(tf.random_normal([])))
qp = PointMass(params=qp_params)

inference = ed.MAP({p: qp}, data={x: x_data})
inference.run(n_iter=50)
