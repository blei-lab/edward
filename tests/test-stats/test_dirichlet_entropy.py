from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import dirichlet
from scipy import stats


def dirichlet_entropy_vec(alpha):
  """Vectorized version of stats.dirichlet.entropy."""
  if len(alpha.shape) == 1:
    return stats.dirichlet.entropy(alpha)
  else:
    n_minibatch = alpha.shape[0]
    return np.array([stats.dirichlet.entropy(alpha[i, :])
                     for i in range(n_minibatch)])


class test_dirichlet_entropy_class(tf.test.TestCase):

  def _test(self, alpha):
    val_true = dirichlet_entropy_vec(alpha)
    with self.test_session():
      self.assertAllClose(dirichlet.entropy(alpha=alpha).eval(), val_true)

  def test_1d(self):
    self._test(alpha=np.array([0.5, 0.5]))
    self._test(alpha=np.array([0.5, 0.5]))

    self._test(alpha=np.array([1.0, 1.0]))
    self._test(alpha=np.array([1.0, 1.0]))

    self._test(alpha=np.array([0.5, 5.0]))
    self._test(alpha=np.array([0.5, 5.0]))

    self._test(alpha=np.array([5.0, 0.5]))
    self._test(alpha=np.array([5.0, 0.5]))

  def test_2d(self):
    self._test(np.array([[0.3, 0.7], [0.5, 0.5]]))
    self._test(np.array([[0.2, 0.8], [0.3, 0.7]]))

if __name__ == '__main__':
  tf.test.main()
