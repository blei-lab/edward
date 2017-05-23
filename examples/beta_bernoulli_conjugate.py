"""A simple coin flipping example that exploits conjugacy.

Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import Bernoulli, Beta

ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
pi = Beta(1.0, 1.0)
x = Bernoulli(probs=pi, sample_shape=10)

# COMPLETE CONDITIONAL
pi_cond = ed.complete_conditional(pi)

sess = ed.get_session()
tf.global_variables_initializer().run()

print('p(pi | x) type:', pi_cond.parameters['name'])
param_vals = sess.run({key: val for
                       key, val in six.iteritems(pi_cond.parameters)
                       if isinstance(val, tf.Tensor)}, {x: x_data})
print('parameters:')
for key, val in six.iteritems(param_vals):
  print('%s:\t%.3f' % (key, val))
