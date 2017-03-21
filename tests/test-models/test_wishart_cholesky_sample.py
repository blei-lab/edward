from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import WishartCholesky


def _test(df, scale, n):
  x = WishartCholesky(df=df, scale=scale)
  val_est = x.sample(n).shape.as_list()
  val_true = n + tf.convert_to_tensor(scale).shape.as_list()
  assert val_est == val_true


class test_wishart_cholesky_sample_class(tf.test.TestCase):

  def test_2d(self):
    with self.test_session():
      df = 5
      scale = tf.diag(tf.ones(3))
      _test(df, scale, [1])
      _test(df, scale, [5])

if __name__ == '__main__':
  tf.test.main()
