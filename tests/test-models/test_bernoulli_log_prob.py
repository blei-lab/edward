from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli
from scipy import stats


def _test(p, n):
  rv = Bernoulli(p=p)
  rv_sample = rv.sample(n)
  x = rv_sample.eval()
  x_tf = tf.constant(x, dtype=tf.float32)
  p = p.eval()
  assert np.allclose(rv.log_prob(x_tf).eval(),
                     stats.bernoulli.logpmf(x, p))


class test_bernoulli_log_prob_class(tf.test.TestCase):

  def test_1d(self):
    ed.set_seed(98765)
    with self.test_session():
      _test(tf.zeros([1]) + 0.5, [1])
      _test(tf.zeros([1]) + 0.5, [5])
      _test(tf.zeros([5]) + 0.5, [1])
      _test(tf.zeros([5]) + 0.5, [5])

if __name__ == '__main__':
  tf.test.main()
