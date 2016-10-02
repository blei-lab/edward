from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Normal, TransformedDistribution
from edward.util import get_dims


def _test(base_dist_cls, transform, inverse,
          log_det_jacobian, n, **base_dist_args):
  x = TransformedDistribution(
      base_dist_cls=base_dist_cls,
      transform=transform,
      inverse=inverse,
      log_det_jacobian=log_det_jacobian,
      **base_dist_args)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(base_dist_args['mu'])
  assert val_est == val_true


class test_transformed_distribution_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      # log-normal
      base_dist_cls = Normal

      def transform(x):
        return tf.sigmoid(x)

      def inverse(y):
        return tf.log(y) - tf.log(1. - y)

      def log_det_jacobian(x):
        return tf.reduce_sum(
            tf.log(tf.sigmoid(x)) + tf.log(1. - tf.sigmoid(x)),
            reduction_indices=[-1])

      mu = 0.0
      sigma = 1.0
      _test(base_dist_cls, transform, inverse,
            log_det_jacobian, [1], mu=mu, sigma=sigma)
      _test(base_dist_cls, transform, inverse,
            log_det_jacobian, [5], mu=mu, sigma=sigma)

  def test_1d(self):
    with self.test_session():
      # log-normal
      base_dist_cls = Normal

      def transform(x):
        return tf.sigmoid(x)

      def inverse(y):
        return tf.log(y) - tf.log(1. - y)

      def log_det_jacobian(x):
        return tf.reduce_sum(
            tf.log(tf.sigmoid(x)) + tf.log(1. - tf.sigmoid(x)),
            reduction_indices=[-1])

      mu = tf.zeros(5)
      sigma = tf.ones(5)
      _test(base_dist_cls, transform, inverse,
            log_det_jacobian, [1], mu=mu, sigma=sigma)
      _test(base_dist_cls, transform, inverse,
            log_det_jacobian, [5], mu=mu, sigma=sigma)

if __name__ == '__main__':
  tf.test.main()
