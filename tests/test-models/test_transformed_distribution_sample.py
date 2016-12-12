from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Normal, TransformedDistribution
from edward.util import get_dims


def _test(distribution, bijector, n):
  x = TransformedDistribution(
      distribution=distribution,
      bijector=bijector,
      validate_args=True)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(distribution.mean())
  assert val_est == val_true


class test_transformed_distribution_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      # log-normal
      mu = 0.0
      sigma = 1.0
      distribution = Normal(mu=mu, sigma=sigma)

      _test(distribution, tf.contrib.distributions.bijector.Exp(), [1])
      _test(distribution, tf.contrib.distributions.bijector.Exp(), [5])

  def test_1d(self):
    with self.test_session():
      # log-normal
      mu = tf.zeros(5)
      sigma = tf.ones(5)
      distribution = Normal(mu=mu, sigma=sigma)

      _test(distribution, tf.contrib.distributions.bijector.Exp(), [1])
      _test(distribution, tf.contrib.distributions.bijector.Exp(), [5])


if __name__ == '__main__':
  tf.test.main()
