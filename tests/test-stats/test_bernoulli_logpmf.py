from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import bernoulli
from scipy import stats


class test_bernoulli_logpmf_class(tf.test.TestCase):

  def _test(self, x, p):
    val_true = stats.bernoulli.logpmf(x, p=p)
    with self.test_session():
      self.assertAllClose(bernoulli.logpmf(x, p=p).eval(), val_true)

  def test_int_0d(self):
    self._test(0, 0.5)
    self._test(1, 0.75)

  def test_float_0d(self):
    self._test(0.0, 0.5)
    self._test(1.0, 0.75)

  def test_int_1d(self):
    self._test([0, 1, 0], 0.5)
    self._test([1, 0, 0], 0.75)

  def test_float_1d(self):
    self._test([0.0, 1.0, 0.0], 0.5)
    self._test([1.0, 0.0, 0.0], 0.75)

  def test_int_2d(self):
    self._test(np.array([[0, 1, 0], [0, 1, 0]]), 0.5)
    self._test(np.array([[1, 0, 0], [0, 1, 0]]), 0.75)

  def test_float_2d(self):
    self._test(np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), 0.5)
    self._test(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.75)

if __name__ == '__main__':
  tf.test.main()
