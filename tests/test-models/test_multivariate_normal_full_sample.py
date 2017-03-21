from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import MultivariateNormalFull


def _test(mu, sigma, n):
  x = MultivariateNormalFull(mu=mu, sigma=sigma)
  val_est = x.sample(n).shape.as_list()
  val_true = n + tf.convert_to_tensor(mu).shape.as_list()
  assert val_est == val_true


class test_multivariate_normal_full_sample_class(tf.test.TestCase):

  def test_1d(self):
    with self.test_session():
      _test(tf.constant([0.5, 0.5]), tf.diag(tf.ones(2)), [1])
      _test(tf.constant([0.5, 0.5]), tf.diag(tf.ones(2)), [5])

if __name__ == '__main__':
  tf.test.main()
