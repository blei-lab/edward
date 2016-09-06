from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import t
from scipy import stats


class test_t_rvs_class(tf.test.TestCase):

  def _test(self, df, loc, scale, size):
    val_est = t.rvs(df, loc, scale, size=size).shape
    val_true = (size, ) + np.asarray(df).shape
    assert val_est == val_true

  def test_0d(self):
    self._test(3, 0.5, 0.5, 1)
    self._test(np.array(3), np.array(0.5), np.array(0.5), 1)

  def test_1d(self):
    self._test(np.array([3]), np.array([0.5]), np.array([0.5]), 1)
    self._test(np.array([3]), np.array([0.5]), np.array([0.5]), 5)
    self._test(np.array([3, 2]), np.array([0.2, 0.8]), np.array([0.2, 0.8]), 1)
    self._test(np.array([3, 2]), np.array([0.2, 0.8]), np.array([0.2, 0.8]), 10)

if __name__ == '__main__':
  tf.test.main()
