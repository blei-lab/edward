from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Bernoulli
from tensorflow.contrib import distributions as ds


class test_bernoulli_sample_class(tf.test.TestCase):

  def _test(self, probs, n):
    rv = Bernoulli(probs)
    dist = ds.Bernoulli(probs)
    self.assertEqual(rv.sample(n).shape, dist.sample(n).shape)

  def test_0d(self):
    with self.test_session():
      self._test(0.5, [1])
      self._test(np.array(0.5), [1])
      self._test(tf.constant(0.5), [1])

  def test_1d(self):
    with self.test_session():
      self._test(np.array([0.5]), [1])
      self._test(np.array([0.5]), [5])
      self._test(np.array([0.2, 0.8]), [1])
      self._test(np.array([0.2, 0.8]), [10])
      self._test(tf.constant([0.5]), [1])
      self._test(tf.constant([0.5]), [5])
      self._test(tf.constant([0.2, 0.8]), [1])
      self._test(tf.constant([0.2, 0.8]), [10])

if __name__ == '__main__':
  tf.test.main()
