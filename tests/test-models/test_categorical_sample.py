from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Categorical
from edward.util import get_dims


def _test(logits, n):
  x = Categorical(logits=logits)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(logits)[:-1]
  assert val_est == val_true


class test_categorical_sample_class(tf.test.TestCase):

  def test_0d(self):
    with self.test_session():
      _test(np.array([0.4, 0.6]), [1])
      _test(np.array([0.4, 0.6]), [5])
      _test(tf.constant([0.4, 0.6]), [5])

if __name__ == '__main__':
  tf.test.main()
