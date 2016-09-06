from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal
from scipy import stats


class test_multivariate_normal_rvs_class(tf.test.TestCase):

  def _test(self, mean, cov, size):
    val_est = multivariate_normal.rvs(mean, cov, size=size).shape
    val_true = (size, ) + np.asarray(mean).shape
    assert val_est == val_true

  def test_1d(self):
    self._test(np.array([0.5]), np.diag([1.0]), 1)
    self._test(np.array([0.5]), np.diag([1.0]), 5)
    self._test(np.array([0.2, 0.8]), np.diag([1.0, 1.0]), 1)
    self._test(np.array([0.2, 0.8]), np.diag([1.0, 1.0]), 10)

  # def test_2d(self):
  #    self._test(np.array([[0.5]]), np.asarray([np.diag([1.0])]), 1)
  #    self._test(np.array([[0.5]]), np.asarray([np.diag([1.0])]), 5)
  #    self._test(np.array([[0.2, 0.8]]), np.asarray([np.diag([1.0]*2)]), 1)
  #    self._test(np.array([[0.2, 0.8]]), np.asarray([np.diag([1.0]*2)]), 10)
  #    self._test(np.array([[0.2, 0.8], [0.7, 0.6]]),
  #                   np.asarray([np.diag([1.0]*2)]*2), 1)
  #    self._test(np.array([[0.2, 0.8], [0.7, 0.6]]),
  #                   np.asarray([np.diag([1.0]*2)]*2), 10)

if __name__ == '__main__':
  tf.test.main()
