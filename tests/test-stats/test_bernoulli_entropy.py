from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import bernoulli
from scipy import stats


class test_bernoulli_entropy_class(tf.test.TestCase):

  def _test(self, p):
    val_true = stats.bernoulli.entropy(p=p)
    self.assertAllClose(bernoulli.entropy(p=p).eval(), val_true)

  def test_0d(self):
    with self.test_session():
      self._test(0.5)
      self._test(0.75)

  def test_1d(self):
    with self.test_session():
      self._test([0.1, 0.9, 0.1])
      self._test([0.5, 0.75, 0.2])

if __name__ == '__main__':
  tf.test.main()
