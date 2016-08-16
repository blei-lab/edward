from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Dirichlet
from edward.util import get_dims


def _test(shape, alpha, n):
  x = Dirichlet(shape, alpha)
  val_est = tuple(get_dims(x.sample(n)))
  val_true = (n, ) + shape
  assert val_est == val_true

class test_dirichlet_sample_class(tf.test.TestCase):

  def test_1d(self):
    with self.test_session():
      _test((2, ), np.array([0.2, 0.8]), 1)
      _test((2, ), np.array([0.2, 0.8]), 10)
      _test((3, ), np.array([0.2, 1.1, 0.8]), 1)
      _test((3, ), np.array([0.2, 1.1, 0.8]), 10)
      _test((2, ), tf.constant([0.2, 0.8]), 1)
      _test((2, ), tf.constant([0.2, 0.8]), 10)
      _test((3, ), tf.constant([0.2, 1.1, 0.8]), 1)
      _test((3, ), tf.constant([0.2, 1.1, 0.8]), 10)
