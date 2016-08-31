from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import PointMass
from edward.util import get_dims
from scipy import stats


def pointmass_logpmf_vec(x, params):
  """Vectorized log-density for point mass distribution."""
  return np.equal(x, params).astype(np.float32)


def _test(params, n):
  rv = PointMass(params=params)
  rv_sample = rv.sample(n)
  x = rv_sample.eval()
  x_tf = tf.constant(x, dtype=tf.float32)
  params = params.eval()
  assert np.allclose(rv.log_prob(x_tf).eval(),
                     pointmass_logpmf_vec(x, params))


class test_pointmass_log_prob_class(tf.test.TestCase):

  def test_1d(self):
    ed.set_seed(98765)
    with self.test_session():
      _test(tf.zeros([1]) + 0.5, [1])
      _test(tf.zeros([1]) + 0.5, [5])
      _test(tf.zeros([5]) + 0.5, [1])
      _test(tf.zeros([5]) + 0.5, [5])

if __name__ == '__main__':
  tf.test.main()
