from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import RandomVariable

from edward.models import *
from edward.util import get_dims


def _test(RV, value, *args, **kwargs):
  rv = RV(*args, value=value, **kwargs)
  value_shape = rv._value.get_shape().as_list()
  expected_shape = (rv.get_batch_shape().as_list() +
                    rv.get_event_shape().as_list())
  assert value_shape == 0
  assert rv.dtype == rv._value.dtype


class test_random_variable_init_value_class(tf.test.TestCase):
  def check_shape_and_dtype(self):
    with self.test_session():
      _test(Normal, 2, mu=0.5, sigma=1.)
      _test(Normal, [2], mu=[0.5], sigma=[1.])
      _test(Poisson, 2, lam=0.5)

  def check_mismatch_raises(self):
    with self.test_session():
        with pytest.raises(ValueError):
          _test(Normal, 2, mu=[0.5, 0.5], sigma=1.)
          _test(Normal, 2, mu=[0.5], sigma=[1.])
          _test(Normal, [2], mu=0.5, sigma=1.)

if __name__ == '__main__':
  tf.test.main()
