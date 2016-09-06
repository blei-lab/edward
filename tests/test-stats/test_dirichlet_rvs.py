from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import dirichlet
from scipy import stats


class test_dirichlet_rvs_class(tf.test.TestCase):

  def _test(self, alpha, size):
    val_est = dirichlet.rvs(alpha, size=size).shape
    val_true = (size, ) + np.asarray(alpha).shape
    assert val_est == val_true

  def test_1d(self):
    self._test(np.array([0.2, 0.8]), 1)
    self._test(np.array([0.2, 0.8]), 10)
    self._test(np.array([0.2, 1.1, 0.8]), 1)
    self._test(np.array([0.2, 1.1, 0.8]), 10)

  # def test_2d(self):
  #    self._test(np.array([[0.2, 0.8]]), 1)
  #    self._test(np.array([[0.2, 0.8]]), 10)
  #    self._test(np.array([[0.2, 1.1, 0.8], [0.7, 0.65, 0.6]]), 1)
  #    self._test(np.array([[0.2, 1.1, 0.8], [0.7, 0.65, 0.6]]), 10)

if __name__ == '__main__':
  tf.test.main()
