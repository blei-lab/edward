from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import geom
from scipy import stats


class test_geom_logpmf_class(tf.test.TestCase):

  def _test(self, x, p):
    val_true = stats.geom.logpmf(x, p)
    with self.test_session():
      self.assertAllClose(geom.logpmf(x, p=p).eval(), val_true)

  def test_int_0d(self):
    self._test(1, 0.5)
    self._test(2, 0.75)

  def test_float_0d(self):
    self._test(1.0, 0.5)
    self._test(2.0, 0.75)

  def test_int_1d(self):
    self._test([1, 5, 3], 0.5)
    self._test([2, 8, 2], 0.75)

  def test_float_1d(self):
    self._test([1.0, 5.0, 3.0], 0.5)
    self._test([2.0, 8.0, 2.0], 0.75)

  def test_int_2d(self):
    self._test(np.array([[1, 5, 3], [2, 8, 2]]), 0.5)
    self._test(np.array([[2, 8, 2], [1, 5, 3]]), 0.75)

  def test_float_2d(self):
    self._test(np.array([[1.0, 5.0, 3.0], [2.0, 8.0, 2.0]]), 0.5)
    self._test(np.array([[2.0, 8.0, 2.0], [1.0, 5.0, 3.0]]), 0.75)

if __name__ == '__main__':
  tf.test.main()
