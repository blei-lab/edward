from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import dirichlet
from scipy import stats


def dirichlet_logpdf_vec(x, alpha):
  """Vectorized version of stats.dirichlet.logpdf."""
  if len(x.shape) == 1:
    return stats.dirichlet.logpdf(x, alpha)
  else:
    size = x.shape[0]
    return np.array([stats.dirichlet.logpdf(x[i, :], alpha)
                     for i in range(size)])


class test_dirichlet_logpdf_class(tf.test.TestCase):

  def _test(self, x, alpha):
    val_true = dirichlet_logpdf_vec(x, alpha)
    with self.test_session():
      self.assertAllClose(dirichlet.logpdf(x, alpha=alpha).eval(), val_true)

  def test_1d(self):
    self._test(np.array([0.3, 0.7]), alpha=np.array([0.5, 0.5]))
    self._test(np.array([0.2, 0.8]), alpha=np.array([0.5, 0.5]))

    self._test(np.array([0.3, 0.7]), alpha=np.array([1.0, 1.0]))
    self._test(np.array([0.2, 0.8]), alpha=np.array([1.0, 1.0]))

    self._test(np.array([0.3, 0.7]), alpha=np.array([0.5, 5.0]))
    self._test(np.array([0.2, 0.8]), alpha=np.array([0.5, 5.0]))

    self._test(np.array([0.3, 0.7]), alpha=np.array([5.0, 0.5]))
    self._test(np.array([0.2, 0.8]), alpha=np.array([5.0, 0.5]))

  def test_2d(self):
    self._test(np.array([[0.3, 0.7], [0.2, 0.8]]), alpha=np.array([0.5, 0.5]))
    self._test(np.array([[0.2, 0.8], [0.3, 0.7]]), alpha=np.array([0.5, 0.5]))

    self._test(np.array([[0.3, 0.7], [0.2, 0.8]]), alpha=np.array([1.0, 1.0]))
    self._test(np.array([[0.2, 0.8], [0.3, 0.7]]), alpha=np.array([1.0, 1.0]))

    self._test(np.array([[0.3, 0.7], [0.2, 0.8]]), alpha=np.array([0.5, 5.0]))
    self._test(np.array([[0.2, 0.8], [0.3, 0.7]]), alpha=np.array([0.5, 5.0]))

    self._test(np.array([[0.3, 0.7], [0.2, 0.8]]), alpha=np.array([5.0, 0.5]))
    self._test(np.array([[0.2, 0.8], [0.3, 0.7]]), alpha=np.array([5.0, 0.5]))

if __name__ == '__main__':
  tf.test.main()
