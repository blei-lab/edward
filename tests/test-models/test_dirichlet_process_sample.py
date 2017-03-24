from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import DirichletProcess, Normal


class test_dirichletprocess_sample_class(tf.test.TestCase):

  def _test(self, n, alpha, base):
    x = DirichletProcess(alpha=alpha, base=base)
    val_est = x.sample(n).shape.as_list()
    val_true = n + tf.convert_to_tensor(alpha).shape.as_list() + \
        tf.convert_to_tensor(base).shape.as_list()
    self.assertEqual(val_est, val_true)

  def test_alpha_0d_base_0d(self):
    with self.test_session():
      self._test([1], 0.5, Normal(mu=0.0, sigma=0.5))
      self._test([5], tf.constant(0.5), Normal(mu=0.0, sigma=0.5))

  def test_alpha_1d_base0d(self):
    with self.test_session():
      self._test([1], np.array([0.5]), Normal(mu=0.0, sigma=0.5))
      self._test([5], tf.constant([0.5]), Normal(mu=0.0, sigma=0.5))
      self._test([1], tf.constant([0.2, 1.5]), Normal(mu=0.0, sigma=0.5))
      self._test([5], tf.constant([0.2, 1.5]), Normal(mu=0.0, sigma=0.5))

  def test_alpha_0d_base1d(self):
    with self.test_session():
      self._test([1], 0.5, Normal(mu=tf.zeros(3), sigma=tf.ones(3)))
      self._test([5], tf.constant(0.5),
                 Normal(mu=tf.zeros(3), sigma=tf.ones(3)))

  def test_alpha_1d_base2d(self):
    with self.test_session():
      self._test([1], np.array([0.5]),
                 Normal(mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4])))
      self._test([5], tf.constant([0.5]),
                 Normal(mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4])))
      self._test([1], tf.constant([0.2, 1.5]),
                 Normal(mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4])))
      self._test([5], tf.constant([0.2, 1.5]),
                 Normal(mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4])))

  def test_persistent_state(self):
    with self.test_session() as sess:
      dp = DirichletProcess(0.1, Normal(mu=0.0, sigma=1.0))
      x = dp.sample(5)
      y = dp.sample(5)
      x_data, y_data, theta = sess.run([x, y, dp.theta])
      for sample in x_data:
        self.assertTrue(sample in theta)
      for sample in y_data:
        self.assertTrue(sample in theta)

if __name__ == '__main__':
  tf.test.main()
