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
        rv.cat.get_batch_shape()).concatenate(rv.components.get_event_shape())
    self.assertEqual(val_est, val_true)

    self.assertEqual(rv.get_sample_shape(), rv.cat.get_sample_shape())
    self.assertEqual(rv.get_sample_shape(), rv.components.get_sample_shape())
    self.assertEqual(rv.get_batch_shape(), rv.cat.get_batch_shape())
    self.assertEqual(rv.get_event_shape(), rv.components.get_event_shape())

  def test_batch_0d_event_0d(self):
    """Mixture of 3 normal distributions."""
    with self.test_session():
      pi = np.array([0.2, 0.3, 0.5], np.float32)
      mu = np.array([1.0, 5.0, 7.0], np.float32)
      sigma = np.array([1.5, 1.5, 1.5], np.float32)

      self._test([], pi, {'mu': mu, 'sigma': sigma}, Normal)
      self._test([5], pi, {'mu': mu, 'sigma': sigma}, Normal)

  def test_batch_0d_event_1d(self):
    """Mixture of 2 Dirichlet distributions."""
    with self.test_session():
      pi = np.array([0.4, 0.6], np.float32)
      alpha = np.ones([2, 3], np.float32)

      self._test([], pi, {'alpha': alpha}, Dirichlet)
      self._test([5], pi, {'alpha': alpha}, Dirichlet)

  def test_batch_1d_event_0d(self):
    """Two mixtures each of 3 beta distributions."""
    with self.test_session():
      pi = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], np.float32)
      a = np.array([[2.0, 0.5], [1.0, 1.0], [0.5, 2.0]], np.float32)
      b = a + 2.0

      self._test([], pi, {'a': a, 'b': b}, Beta)
      self._test([5], pi, {'a': a, 'b': b}, Beta)

      pi = np.array([0.2, 0.3, 0.5], np.float32)
      self.assertRaises(ValueError, self._test, [], pi, {'a': a, 'b': b}, Beta)

if __name__ == '__main__':
  tf.test.main()
