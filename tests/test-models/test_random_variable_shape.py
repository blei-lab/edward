from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Dirichlet


class test_random_variable_shape_class(tf.test.TestCase):

  def _test_sample(self, rv, sample_shape, batch_shape, event_shape):
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.get_sample_shape(), sample_shape)
    self.assertEqual(rv.get_batch_shape(), batch_shape)
    self.assertEqual(rv.get_event_shape(), event_shape)

  def test_bernoulli(self):
    with self.test_session():
      self._test_sample(Bernoulli(0.5), [], [], [])
      self._test_sample(Bernoulli(tf.zeros([2, 3])), [], [2, 3], [])
      self._test_sample(Bernoulli(0.5, sample_shape=2), [2], [], [])
      self._test_sample(Bernoulli(0.5, sample_shape=[2, 1]), [2, 1], [], [])

  def test_dirichlet(self):
    with self.test_session():
      self._test_sample(Dirichlet(tf.zeros(3)), [], [], [3])
      self._test_sample(Dirichlet(tf.zeros([2, 3])), [], [2], [3])
      self._test_sample(Dirichlet(tf.zeros(3), sample_shape=1), [1], [], [3])
      self._test_sample(Dirichlet(tf.zeros(3), sample_shape=[2, 1]),
                        [2, 1], [], [3])

if __name__ == '__main__':
  tf.test.main()
