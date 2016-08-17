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

  def _test(self, loc=0, scale=1):
    val_true = uniform_entropy_vec(loc, scale)
    with self.test_session():
      self.assertAllClose(uniform.entropy(loc, scale).eval(), val_true)
      self.assertAllClose(uniform.entropy(tf.constant(loc),
                                          tf.constant(scale)).eval(), val_true)

  def test_0d(self):
    self._test()
    self._test(loc=1.0, scale=1.0)
    self._test(loc=0.5, scale=5.0)
    self._test(loc=5.0, scale=0.5)

  def test_1d(self):
    self._test([0.5, 0.3, 0.8, 0.2], [0.5, 0.3, 0.8, 0.2])
