from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Beta, Dirichlet, Normal, ParamMixture


class test_param_mixture_sample_class(tf.test.TestCase):

  def _test(self, n, *args, **kwargs):
    rv = ParamMixture(*args, **kwargs)
    val_est = rv.sample(n).shape
    val_true = tf.TensorShape(n).concatenate(
        rv.cat.batch_shape).concatenate(rv.components.event_shape)
    self.assertEqual(val_est, val_true)

    self.assertEqual(rv.sample_shape, rv.cat.sample_shape)
    self.assertEqual(rv.sample_shape, rv.components.sample_shape)
    self.assertEqual(rv.batch_shape, rv.cat.batch_shape)
    self.assertEqual(rv.event_shape, rv.components.event_shape)

  def test_batch_0d_event_0d(self):
    """Mixture of 3 normal distributions."""
    with self.test_session():
      probs = np.array([0.2, 0.3, 0.5], np.float32)
      loc = np.array([1.0, 5.0, 7.0], np.float32)
      scale = np.array([1.5, 1.5, 1.5], np.float32)

      self._test([], probs, {'loc': loc, 'scale': scale}, Normal)
      self._test([5], probs, {'loc': loc, 'scale': scale}, Normal)

  def test_batch_0d_event_1d(self):
    """Mixture of 2 Dirichlet distributions."""
    with self.test_session():
      probs = np.array([0.4, 0.6], np.float32)
      concentration = np.ones([2, 3], np.float32)

      self._test([], probs, {'concentration': concentration}, Dirichlet)
      self._test([5], probs, {'concentration': concentration}, Dirichlet)

  def test_batch_1d_event_0d(self):
    """Two mixtures each of 3 beta distributions."""
    with self.test_session():
      probs = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], np.float32)
      conc1 = np.array([[2.0, 0.5], [1.0, 1.0], [0.5, 2.0]], np.float32)
      conc0 = conc1 + 2.0

      self._test([], probs, {'concentration1': conc1, 'concentration0': conc0},
                 Beta)
      self._test([5], probs, {'concentration1': conc1, 'concentration0': conc0},
                 Beta)

      probs = np.array([0.2, 0.3, 0.5], np.float32)
      self.assertRaises(ValueError, self._test, [], probs,
                        {'concentration1': conc1, 'concentration0': conc0},
                        Beta)

if __name__ == '__main__':
  tf.test.main()
