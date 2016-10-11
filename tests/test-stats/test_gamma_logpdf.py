from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import gamma
from scipy import stats


class test_gamma_logpdf_class(tf.test.TestCase):

  def _test(self, x, alpha, beta):
    val_true = stats.gamma.logpdf(x, alpha, scale=1.0 / beta)
    with self.test_session():
      self.assertAllClose(gamma.logpdf(x, alpha=alpha, beta=beta).eval(),
                          val_true)

  def test_0d(self):
    self._test(0.3, alpha=0.5, beta=1.0)
    self._test(0.7, alpha=0.5, beta=1.0)

    self._test(0.3, alpha=1.0, beta=1.0)
    self._test(0.7, alpha=1.0, beta=1.0)

    self._test(0.3, alpha=0.5, beta=5.0)
    self._test(0.7, alpha=0.5, beta=5.0)

    self._test(0.3, alpha=5.0, beta=0.5)
    self._test(0.7, alpha=5.0, beta=0.5)

  def test_1d(self):
    self._test([0.5, 1.2, 5.3, 8.7], alpha=0.5, beta=0.5)

  def test_2d(self):
    self._test(np.array([[0.5, 1.2, 5.3, 8.7], [0.5, 1.2, 5.3, 8.7]],
                        dtype=np.float32),
               alpha=0.5, beta=0.5)

if __name__ == '__main__':
  tf.test.main()
