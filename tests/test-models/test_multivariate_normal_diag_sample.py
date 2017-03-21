from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import MultivariateNormalDiag


def _test(mu, diag_stdev, n):
  x = MultivariateNormalDiag(mu=mu, diag_stdev=diag_stdev)
  val_est = x.sample(n).shape.as_list()
  val_true = n + tf.convert_to_tensor(mu).shape.as_list()
  assert val_est == val_true


class test_multivariate_normal_diag_sample_class(tf.test.TestCase):

  def test_1d(self):
    with self.test_session():
      _test(tf.constant([0.5, 0.5]), tf.ones(2), [1])
      _test(tf.constant([0.5, 0.5]), tf.ones(2), [5])

if __name__ == '__main__':
  tf.test.main()
