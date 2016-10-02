from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import MultivariateNormalDiagPlusVDVT
from edward.util import get_dims


def _test(mu, diag_large, v, diag_small, n):
  x = MultivariateNormalDiagPlusVDVT(
      mu=mu, diag_large=diag_large, v=v, diag_small=diag_small)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(mu)
  assert val_est == val_true


class test_multivariate_normal_diag_plus_vdvt_sample_class(tf.test.TestCase):

  def test_1d(self):
    with self.test_session():
      _test(tf.constant([0.5, 0.5]), tf.ones(2),
            tf.diag(tf.ones(2)), tf.ones(2), [1])
      _test(tf.constant([0.5, 0.5]), tf.ones(2),
            tf.diag(tf.ones(2)), tf.ones(2), [5])

if __name__ == '__main__':
  tf.test.main()
