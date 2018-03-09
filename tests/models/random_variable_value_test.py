from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Poisson, RandomVariable


class test_random_variable_value_class(tf.test.TestCase):

  def _test_sample(self, RV, value, *args, **kwargs):
    rv = RV(*args, value=value, **kwargs)
    value_shape = rv.value.shape
    expected_shape = rv.sample_shape.concatenate(
        rv.batch_shape).concatenate(rv.event_shape)
    self.assertEqual(value_shape, expected_shape)
    self.assertEqual(rv.dtype, rv.value.dtype)

  def test_shape_and_dtype(self):
    with self.test_session():
      self._test_sample(Normal, 2, loc=0.5, scale=1.0)
      self._test_sample(Normal, [2], loc=[0.5], scale=[1.0])
      self._test_sample(Poisson, 2, rate=0.5)

  def test_unknown_shape(self):
    with self.test_session():
      x = Bernoulli(0.5, value=tf.placeholder(tf.int32))

  def test_mismatch_raises(self):
    with self.test_session():
      self.assertRaises(ValueError, self._test_sample, Normal, 2,
                        loc=[0.5, 0.5], scale=1.0)
      self.assertRaises(ValueError, self._test_sample, Normal, 2,
                        loc=[0.5], scale=[1.0])
      self.assertRaises(ValueError, self._test_sample, Normal,
                        np.zeros([10, 3]), loc=[0.5, 0.5], scale=[1.0, 1.0])

if __name__ == '__main__':
  tf.test.main()
