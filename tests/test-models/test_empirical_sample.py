from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Empirical


def _test(params, n):
  x = Empirical(params=params)
  val_est = x.sample(n).shape.as_list()
  val_true = n + tf.convert_to_tensor(params).shape.as_list()[1:]
  assert val_est == val_true


class test_empirical_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      _test(0.5, [1])
      _test(np.array(0.5), [1])
      _test(tf.constant(0.5), [1])
      _test(np.array([0.5]), [1])
      _test(np.array([0.5]), [5])
      _test(np.array([0.2, 0.8]), [1])
      _test(np.array([0.2, 0.8]), [10])
      _test(tf.constant([0.5]), [1])
      _test(tf.constant([0.5]), [5])
      _test(tf.constant([0.2, 0.8]), [1])
      _test(tf.constant([0.2, 0.8]), [10])

if __name__ == '__main__':
  tf.test.main()
