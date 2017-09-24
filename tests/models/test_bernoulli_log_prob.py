from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli
from tensorflow.contrib import distributions as ds


class test_bernoulli_log_prob_class(tf.test.TestCase):

  def _test(self, probs, n):
    rv = Bernoulli(probs)
    dist = ds.Bernoulli(probs)
    x = rv.sample(n).eval()
    self.assertAllEqual(rv.log_prob(x).eval(), dist.log_prob(x).eval())

  def test_1d(self):
    with self.test_session():
      self._test(tf.zeros([1]) + 0.5, [1])
      self._test(tf.zeros([1]) + 0.5, [5])
      self._test(tf.zeros([5]) + 0.5, [1])
      self._test(tf.zeros([5]) + 0.5, [5])

if __name__ == '__main__':
  tf.test.main()
