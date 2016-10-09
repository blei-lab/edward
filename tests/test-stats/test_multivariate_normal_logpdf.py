from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal
from scipy import stats


class test_multivariate_normal_logpdf_class(tf.test.TestCase):

  def _test(self, x, mu, sigma):
    val_true = stats.multivariate_normal.logpdf(x, mu, sigma)
    with self.test_session():
      self.assertAllClose(
          multivariate_normal.logpdf(x, mu=mu, sigma=sigma).eval(), val_true)

  def test_1d(self):
    x = np.array([0.0, 0.0])
    self._test(x, np.zeros(2), np.diag(np.ones(2)))
    self._test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))

  def test_2d(self):
    x = np.array([[0.3, 0.7], [0.2, 0.8]])
    self._test(x, np.zeros(2), np.diag(np.ones(2)))
    self._test(x, np.zeros(2), np.array([[2.0, 0.5], [0.5, 1.0]]))

if __name__ == '__main__':
  tf.test.main()
