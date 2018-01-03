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
loc = 7.0
scale = 0.7
xn_data = np.random.normal(loc, scale, N)
print('scale={}'.format(scale))

# Prior definition
alpha = tf.Variable(0.5, trainable=False)
beta = tf.Variable(0.7, trainable=False)

# Posterior inference
# Probabilistic model
ig = InverseGamma(alpha, beta)
xn = Normal(loc, tf.ones([N]) * tf.sqrt(ig))

# Inference
qig = Empirical(params=tf.Variable(tf.zeros(1000) + 0.5))
proposal_ig = InverseGamma(2.0, 2.0)
inference = ed.MetropolisHastings({ig: qig},
                                  {ig: proposal_ig}, data={xn: xn_data})
inference.run()

sess = ed.get_session()
print('Inferred scale={}'.format(sess.run(tf.sqrt(qig.mean()))))
