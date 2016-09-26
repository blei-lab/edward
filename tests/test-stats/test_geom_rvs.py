from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import geom
from scipy import stats


class test_geom_rvs_class(tf.test.TestCase):

  def _test(self, p, size):
    val_est = geom.rvs(p, size=size).shape
    val_true = (size, ) + np.asarray(p).shape
    assert val_est == val_true

  def test_0d(self):
    self._test(0.5, 1)
    self._test(np.array(0.5), 1)

  def test_1d(self):
    self._test(np.array([0.5]), 1)
    self._test(np.array([0.5]), 5)
    self._test(np.array([0.2, 0.8]), 1)
    self._test(np.array([0.2, 0.8]), 10)

  # def test_2d(self):
  #    self._test(np.array([[0.5]]), 1)
  #    self._test(np.array([[0.5]]), 5)
  #    self._test(np.array([[0.2, 0.8]]), 1)
  #    self._test(np.array([[0.2, 0.8]]), 10)
  #    self._test(np.array([[0.2, 0.8], [0.7, 0.6]]), 1)
  #    self._test(np.array([[0.2, 0.8], [0.7, 0.6]]), 10)

if __name__ == '__main__':
  tf.test.main()
