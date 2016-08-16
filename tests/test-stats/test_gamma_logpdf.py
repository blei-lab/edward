from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import gamma
from scipy import stats

class test_gamma_logpdf_class(tf.test.TestCase):

  def _test(self, x, a, scale=1):
    xtf = tf.constant(x)
    val_true = stats.gamma.logpdf(x, a, scale=scale)
    with self.test_session():
      self.assertAllClose(gamma.logpdf(xtf, tf.constant(a), tf.constant(scale)).eval(), val_true)

  def test_0d(self):
    self._test(0.3, a=0.5)
    self._test(0.7, a=0.5)

    self._test(0.3, a=1.0, scale=1.0)
    self._test(0.7, a=1.0, scale=1.0)

    self._test(0.3, a=0.5, scale=5.0)
    self._test(0.7, a=0.5, scale=5.0)

    self._test(0.3, a=5.0, scale=0.5)
    self._test(0.7, a=5.0, scale=0.5)


  def test_1d(self):
    self._test([0.5, 1.2, 5.3, 8.7], a=0.5, scale=0.5)


  def test_2d(self):
    self._test(np.array([[0.5, 1.2, 5.3, 8.7],[0.5, 1.2, 5.3, 8.7]]),
           a=0.5, scale=0.5)
