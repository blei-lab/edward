from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import beta
from scipy import stats


class test_beta_entropy_class(tf.test.TestCase):

  def _test(self, a, b):
    val_true = stats.beta.entropy(a, b)
    self.assertAllClose(beta.entropy(a=a, b=b).eval(), val_true, atol=1e-4)

  def test_0d(self):
    with self.test_session():
      self._test(a=0.5, b=0.5)
      self._test(a=1.0, b=1.0)
      self._test(a=0.5, b=5.0)
      self._test(a=5.0, b=0.5)

  def test_1d(self):
    with self.test_session():
      self._test([0.5, 0.3, 0.8, 0.1], [0.1, 0.7, 0.2, 0.4])

if __name__ == '__main__':
  tf.test.main()
