from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import chi2
from scipy import stats


class test_chi2_rvs_class(tf.test.TestCase):

  def _test(self, df, size):
    val_est = chi2.rvs(df, size=size).shape
    val_true = (size, ) + np.asarray(df).shape
    assert val_est == val_true

  def test_0d(self):
    self._test(3, 1)
    self._test(np.array(3), 1)

  def test_1d(self):
    self._test(np.array([3]), 1)
    self._test(np.array([3]), 5)
    self._test(np.array([3, 2]), 1)
    self._test(np.array([3, 2]), 10)

  # def test_2d(self):
  #    self._test(np.array([[3]]), 1)
  #    self._test(np.array([[3]]), 5)
  #    self._test(np.array([[3, 2]]), 1)
  #    self._test(np.array([[3, 2]]), 10)
  #    self._test(np.array([[3, 2], [7, 4]]), 1)
  #    self._test(np.array([[3, 2], [7, 4]]), 10)

if __name__ == '__main__':
  tf.test.main()
