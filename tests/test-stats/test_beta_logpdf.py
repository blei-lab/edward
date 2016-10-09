from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import beta
from scipy import stats


class test_beta_logpdf_class(tf.test.TestCase):

  def _test(self, x, a, b):
    val_true = stats.beta.logpdf(x, a, b)
    with self.test_session():
      self.assertAllClose(beta.logpdf(x, a=a, b=b).eval(), val_true)

  def test_0d(self):
    self._test(0.3, a=0.5, b=0.5)
    self._test(0.7, a=0.5, b=0.5)

    self._test(0.3, a=1.0, b=1.0)
    self._test(0.7, a=1.0, b=1.0)

    self._test(0.3, a=0.5, b=5.0)
    self._test(0.7, a=0.5, b=5.0)

    self._test(0.3, a=5.0, b=0.5)
    self._test(0.7, a=5.0, b=0.5)

  def test_1d(self):
    self._test([0.5, 0.3, 0.8, 0.1], a=0.5, b=0.5)

  def test_2d(self):
    self._test(np.array([[0.5, 0.3, 0.8, 0.1], [0.1, 0.7, 0.2, 0.4]],
                        dtype=np.float32),
               a=0.5, b=0.5)

if __name__ == '__main__':
  tf.test.main()
