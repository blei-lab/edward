from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import invgamma
from scipy import stats

def invgamma_entropy_vec(a, scale):
  if isinstance(scale, float):
    return stats.invgamma.entropy(a, scale=scale)
  else:
    return [stats.invgamma.entropy(a_x, scale=scale_x)
        for a_x, scale_x in zip(a, scale)]

class test_invgamma_entropy_class(tf.test.TestCase):

  def _test(self, a, scale=1):
    val_true = invgamma_entropy_vec(a, scale=scale)
    with self.test_session():
      self.assertAllClose(invgamma.entropy(a, scale).eval(), val_true, atol=1e-4)
      self.assertAllClose(invgamma.entropy(tf.constant(a), tf.constant(scale)).eval(), val_true, atol=1e-4)

  def test_0d(self):
    self._test(a=1.0, scale=1.0)
    self._test(a=0.5, scale=5.0)
    self._test(a=5.0, scale=0.5)


  def test_1d(self):
    self._test([0.5, 1.2, 5.3, 8.7], [0.5, 1.2, 5.3, 8.7])
