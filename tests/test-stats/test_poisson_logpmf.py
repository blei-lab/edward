from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import poisson
from scipy import stats


class test_poisson_logpmf_class(tf.test.TestCase):

  def _test(self, x, mu):
    xtf = tf.constant(x)
    val_true = stats.poisson.logpmf(x, mu)
    with self.test_session():
      self.assertAllClose(poisson.logpmf(xtf, mu).eval(), val_true)
      self.assertAllClose(poisson.logpmf(xtf, tf.constant(mu)).eval(), val_true)

  def test_int_0d(self):
    self._test(0, 0.5)
    self._test(1, 0.75)

  def test_float_0d(self):
    self._test(0.0, 0.5)
    self._test(1.0, 0.75)

  def test_int_1d(self):
    self._test([0, 1, 3], 0.5)
    self._test([1, 8, 2], 0.75)

  def test_float_1d(self):
    self._test([0.0, 1.0, 3.0], 0.5)
    self._test([1.0, 8.0, 2.0], 0.75)

  def test_int_2d(self):
    self._test(np.array([[0, 1, 3], [1, 8, 2]]), 0.5)
    self._test(np.array([[1, 8, 2], [0, 1, 3]]), 0.75)

  def test_float_2d(self):
    self._test(np.array([[0.0, 1.0, 3.0], [0.0, 1.0, 3.0]]), 0.5)
    self._test(np.array([[1.0, 8.0, 2.0], [1.0, 8.0, 2.0]]), 0.75)

if __name__ == '__main__':
  tf.test.main()
