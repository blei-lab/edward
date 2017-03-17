from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import DirichletProcess, Normal
from edward.util import get_dims


def _test(n, alpha, base_cls, *args, **kwargs):
  x = DirichletProcess(alpha=alpha, base_cls=base_cls, *args, **kwargs)
  base = base_cls(*args, **kwargs)
  val_est = get_dims(x.sample(n))
  val_true = n + get_dims(alpha) + get_dims(base)
  assert val_est == val_true


class test_dirichletprocess_sample_class(tf.test.TestCase):

  def test_alpha_0d_base_0d(self):
    with self.test_session():
      _test([1], 0.5, Normal, mu=0.0, sigma=0.5)
      _test([5], tf.constant(0.5), Normal, mu=0.0, sigma=0.5)

  def test_alpha_1d_base0d(self):
    with self.test_session():
      _test([1], np.array([0.5]), Normal, mu=0.0, sigma=0.5)
      _test([5], tf.constant([0.5]), Normal, mu=0.0, sigma=0.5)
      _test([1], tf.constant([0.2, 1.5]), Normal, mu=0.0, sigma=0.5)
      _test([5], tf.constant([0.2, 1.5]), Normal, mu=0.0, sigma=0.5)

  def test_alpha_0d_base1d(self):
    with self.test_session():
      _test([1], 0.5, Normal, mu=tf.zeros(3), sigma=tf.ones(3))
      _test([5], tf.constant(0.5), Normal, mu=tf.zeros(3), sigma=tf.ones(3))

  def test_alpha_1d_base2d(self):
    with self.test_session():
      _test([1], np.array([0.5]), Normal,
            mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4]))
      _test([5], tf.constant([0.5]), Normal,
            mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4]))
      _test([1], tf.constant([0.2, 1.5]), Normal,
            mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4]))
      _test([5], tf.constant([0.2, 1.5]), Normal,
            mu=tf.zeros([3, 4]), sigma=tf.ones([3, 4]))

if __name__ == '__main__':
  tf.test.main()
