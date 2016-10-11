from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import norm
from scipy import stats


def norm_entropy_vec(loc, scale):
  """Vectorized version of stats.norm.entropy."""
  if isinstance(loc, float) or isinstance(loc, int):
    return stats.norm.entropy(loc, scale)
  else:
    return np.array([stats.norm.entropy(loc_x, scale_x)
                     for loc_x, scale_x in zip(loc, scale)])


class test_norm_entropy_class(tf.test.TestCase):

  def _test(self, mu, sigma):
    val_true = norm_entropy_vec(mu, sigma)
    with self.test_session():
      self.assertAllClose(norm.entropy(mu=mu, sigma=sigma).eval(), val_true)

  def test_0d(self):
    self._test(1.0, 1.0)
    self._test(1.0, 1.0)

    self._test(0.5, 5.0)
    self._test(5.0, 0.5)

  def test_1d(self):
    self._test([0.5, 1.2, 5.3, 8.7], [0.5, 1.2, 5.3, 8.7])

if __name__ == '__main__':
  tf.test.main()
