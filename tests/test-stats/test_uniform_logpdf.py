from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import uniform
from scipy import stats

class test_uniform_logpdf_class(tf.test.TestCase):

  def _test(self, x, loc=0, scale=1):
    xtf = tf.constant(x)
    val_true = stats.uniform.logpdf(x, loc, scale)
    with self.test_session():
      self.assertAllClose(uniform.logpdf(xtf, loc, scale).eval(), val_true)
      self.assertAllClose(uniform.logpdf(xtf, tf.constant(loc), tf.constant(scale)).eval(), val_true)


  def test_0d(self):
    self._test(0.3)
    self._test(0.7)

    self._test(1.3, loc=1.0, scale=1.0)
    self._test(1.7, loc=1.0, scale=1.0)

    self._test(2.3, loc=0.5, scale=5.0)
    self._test(2.7, loc=0.5, scale=5.0)

    self._test(5.3, loc=5.0, scale=0.5)
    self._test(5.1, loc=5.0, scale=0.5)


  def test_1d(self):
    self._test([0.5, 0.3, 0.8, 0.2], loc=0.1, scale=0.9)


  def test_2d(self):
    self._test(np.array([[0.5, 0.3, 0.8, 0.2],[0.5, 0.3, 0.8, 0.2]]),
           loc=0.1, scale=0.9)
