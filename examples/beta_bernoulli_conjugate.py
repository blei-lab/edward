"""A simple coin flipping example that exploits conjugacy.

Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta

ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
pi = Beta(a=1.0, b=1.0)
x = Bernoulli(p=pi, sample_shape=10)

# COMPLETE CONDITIONAL
pi_cond = ed.complete_conditional(pi)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('p(pi | x) type:', pi_cond.parameters['name'])
relevant_params = {key: val for key, val in pi_cond.parameters.iteritems()
                   if isinstance(val, tf.Tensor)}
param_vals = sess.run(relevant_params.values(), {x: x_data})
print('parameters:')
for i, j in enumerate(relevant_params.keys()):
    print('%s:\t%.3f' % (j, param_vals[i]))
