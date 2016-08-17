from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli
from scipy import stats


def _test(shape, n):
  rv = Bernoulli(shape, p=tf.zeros(shape) + 0.5)
  rv_sample = rv.sample(n)
  x = rv_sample.eval()
  x_tf = tf.constant(x, dtype=tf.float32)
  p = rv.p.eval()
  for idx in range(shape[0]):
    assert np.allclose(rv.log_prob_idx((idx, ), x_tf).eval(),
                       stats.bernoulli.logpmf(x[:, idx], p[idx]))


class test_bernoulli_log_prob_idx_class(tf.test.TestCase):

  def test_1d(self):
    ed.set_seed(98765)
    with self.test_session():
      _test((1, ), 1)
      _test((1, ), 5)
      _test((5, ), 1)
      _test((5, ), 5)
