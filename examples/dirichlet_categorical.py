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

# Data generation
alpha = np.array([20., 30., 10., 10.])
pi = np.random.dirichlet(alpha).astype(np.float32)
zn_data = np.array([np.random.choice(K, 1, p=pi)[0] for n in range(N)])
print('pi={}'.format(pi))

# Prior definition
alpha_prior = tf.Variable(np.array([1., 1., 1., 1.]),
                          dtype=tf.float32, trainable=False)

# Posterior inference
# Probabilistic model
pi = Dirichlet(alpha=alpha_prior)
zn = Categorical(p=tf.ones([N, 1]) * pi)

# Variational model
qpi = Dirichlet(alpha=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# Inference
inference = ed.KLqp({pi: qpi}, data={zn: zn_data})
inference.run(n_iter=1500, n_samples=30)

sess = ed.get_session()
print('Inferred pi={}'.format(sess.run(qpi.mean())))
