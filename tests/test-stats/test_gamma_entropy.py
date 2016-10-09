from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import gamma
from scipy import stats


def gamma_entropy_vec(a, scale):
  """Vectorized version of stats.gamma.entropy."""
  if isinstance(scale, float):
    return stats.gamma.entropy(a, scale=scale)
  else:
    return np.array([stats.gamma.entropy(a_x, scale=scale_x)
                     for a_x, scale_x in zip(a, scale)])


class test_gamma_entropy_class(tf.test.TestCase):

  def _test(self, alpha, beta):
    val_true = gamma_entropy_vec(alpha, scale=1.0 / beta)
    with self.test_session():
      self.assertAllClose(gamma.entropy(alpha=alpha, beta=beta).eval(),
                          val_true, atol=1e-4)

  def test_0d(self):
    self._test(alpha=1.0, beta=1.0)
    self._test(alpha=1.0, beta=1.0)

    self._test(alpha=0.5, beta=5.0)
    self._test(alpha=5.0, beta=0.5)

  def test_1d(self):
    self._test(alpha=np.array([0.5, 1.2, 5.3, 8.7], dtype=np.float32),
               beta=np.array([0.5, 1.2, 5.3, 8.7], dtype=np.float32))

if __name__ == '__main__':
  tf.test.main()
