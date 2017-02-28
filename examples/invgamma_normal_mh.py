#!/usr/bin/env python
""" InverseGamma-Normal model

Posterior inference with Metropolis Hastings
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import InverseGamma, Normal, Empirical

N = 1000

# Data generation (known mean)
mu = 7.0
sigma = 0.7
xn_data = np.random.normal(mu, sigma, N)
print('sigma={}'.format(sigma))

# Prior definition
alpha = tf.Variable(0.5, dtype=tf.float32, trainable=False)
beta = tf.Variable(0.7, dtype=tf.float32, trainable=False)

# Posterior inference
# Probabilistic model
ig = InverseGamma(alpha=alpha, beta=beta)
xn = Normal(mu=mu, sigma=tf.ones([N]) * tf.sqrt(ig))

# Inference
qig = Empirical(params=tf.Variable(tf.zeros(1000) + 0.5))
proposal_ig = InverseGamma(alpha=2.0, beta=2.0)
inference = ed.MetropolisHastings({ig: qig},
                                  {ig: proposal_ig}, data={xn: xn_data})
inference.run()

sess = ed.get_session()
print('Inferred sigma={}'.format(sess.run(tf.sqrt(qig.mean()))))
