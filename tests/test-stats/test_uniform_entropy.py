from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import uniform
from scipy import stats


def uniform_entropy_vec(loc, scale):
  """Vectorized version of stats.uniform.entropy."""
  if isinstance(loc, float) or isinstance(loc, int):
    return stats.uniform.entropy(loc, scale)
  else:
    return np.array([stats.uniform.entropy(loc_x, scale_x)
                     for loc_x, scale_x in zip(loc, scale)])


class test_uniform_entropy_class(tf.test.TestCase):

  def _test(self, a, b):
    val_true = uniform_entropy_vec(a, b - a)
    with self.test_session():
      self.assertAllClose(uniform.entropy(a=a, b=b).eval(), val_true)

  def test_0d(self):
    self._test(a=1.0, b=2.0)
    self._test(a=0.5, b=5.0)
    self._test(a=5.0, b=5.5)

  def test_1d(self):
    self._test(np.array([0.5, 0.3, 0.8, 0.2], dtype=np.float32),
               np.array([0.6, 0.4, 0.9, 0.3], dtype=np.float32))

if __name__ == '__main__':
  tf.test.main()
