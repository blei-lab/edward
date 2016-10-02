from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Dirichlet
from scipy import stats


def dirichlet_logpdf_vec(x, alpha):
  """Vectorized version of stats.dirichlet.logpdf."""
  shape = x.shape
  if len(shape) == 1:
    try:
      return stats.dirichlet.logpdf(x, alpha)
    except:
      x[-1] = 1.0 - np.sum(x[:-1])
      return stats.dirichlet.logpdf(x, alpha)
  elif len(shape) == 2:
    size = shape[0]
    if len(alpha.shape) == 1:
      return np.array([dirichlet_logpdf_vec(x[i, :], alpha)
                       for i in range(size)])
    else:
      return np.array([dirichlet_logpdf_vec(x[i, :], alpha[i, :])
                       for i in range(size)])
  elif len(shape) == 3:
    size = shape[0]
    return np.array([dirichlet_logpdf_vec(x[i, :, :], alpha)
                     for i in range(size)])
  else:
    raise NotImplementedError()


def _test(alpha, n):
  rv = Dirichlet(alpha=alpha)
  rv_sample = rv.sample(n)
  x = rv_sample.eval()
  x_tf = tf.constant(x, dtype=tf.float32)
  alpha = alpha.eval()
  assert np.allclose(rv.log_prob(x_tf).eval(),
                     dirichlet_logpdf_vec(x, alpha), atol=1e-3)


class test_dirichlet_log_prob_class(tf.test.TestCase):

  def test_1d(self):
    ed.set_seed(98765)
    with self.test_session():
      _test(tf.constant([0.6, 0.4]), [1])
      _test(tf.constant([0.6, 0.4]), [2])

  def test_2d(self):
    ed.set_seed(12142)
    with self.test_session():
      _test(tf.constant([[0.5, 0.5], [0.6, 0.4]]), [1])
      _test(tf.constant([[0.5, 0.5], [0.6, 0.4]]), [2])
      _test(tf.constant([[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]]), [1])
      _test(tf.constant([[0.3, 0.2, 0.5], [0.6, 0.1, 0.3]]), [2])

if __name__ == '__main__':
  tf.test.main()
