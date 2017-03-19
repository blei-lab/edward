#!/usr/bin/env python
"""Dirichlet-Categorical model.

Posterior inference with Edward's BBVI.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Dirichlet

N = 1000
K = 4

# DATA
pi_true = np.random.dirichlet(np.array([20.0, 30.0, 10.0, 10.0]))
z_data = np.array([np.random.choice(K, 1, p=pi_true)[0] for n in range(N)])
print('pi={}'.format(pi_true))

# MODEL
pi = Dirichlet(alpha=tf.ones(4))
z = Categorical(p=tf.ones([N, 1]) * pi)

# INFERENCE
qpi = Dirichlet(alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

inference = ed.KLqp({pi: qpi}, data={z: z_data})
inference.run(n_iter=1500, n_samples=30)

sess = ed.get_session()
print('Inferred pi={}'.format(sess.run(qpi.mean())))
