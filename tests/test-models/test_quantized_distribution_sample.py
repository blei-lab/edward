from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Normal, QuantizedDistribution
from edward.util import get_dims


def _test(base_dist_cls, lower_cutoff, upper_cutoff, n, **base_dist_args):
  x = QuantizedDistribution(
      base_dist_cls=base_dist_cls,
      lower_cutoff=lower_cutoff,
      upper_cutoff=upper_cutoff,
      **base_dist_args)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(lower_cutoff)
  assert val_est == val_true


class test_quantized_distribution_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      base_dist_cls = Normal
      lower_cutoff = 0.0
      upper_cutoff = 2.0
      mu = 0.0
      sigma = 1.0
      _test(base_dist_cls, lower_cutoff, upper_cutoff, [1], mu=mu, sigma=sigma)
      _test(base_dist_cls, lower_cutoff, upper_cutoff, [5], mu=mu, sigma=sigma)

  def test_1d(self):
    with self.test_session():
      base_dist_cls = Normal
      lower_cutoff = tf.constant([0.0, 1.0, 2.0, 4.0, 1.0])
      upper_cutoff = tf.constant([2.0, 3.0, 3.0, 6.0, 2.0])
      mu = tf.zeros(5)
      sigma = tf.ones(5)
      _test(base_dist_cls, lower_cutoff, upper_cutoff, [1], mu=mu, sigma=sigma)
      _test(base_dist_cls, lower_cutoff, upper_cutoff, [5], mu=mu, sigma=sigma)

if __name__ == '__main__':
  tf.test.main()
