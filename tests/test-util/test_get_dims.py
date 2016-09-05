from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Bernoulli
from edward.util import get_dims


class test_get_dims_class(tf.test.TestCase):

  def test_get_dims_0d_int(self):
    with self.test_session():
      x = 0
      self.assertEqual(get_dims(x), [])

  def test_get_dims_0d_float(self):
    with self.test_session():
      x = 0.0
      self.assertEqual(get_dims(x), [])

  def test_get_dims_0d_tf(self):
    with self.test_session():
      x = tf.constant(0.0)
      self.assertEqual(get_dims(x), [])

  def test_get_dims_0d_np(self):
    with self.test_session():
      x = np.array(0.0)
      self.assertEqual(get_dims(x), [])

  def test_get_dims_0d_rv(self):
    with self.test_session():
      x = Bernoulli(p=0.5)
      self.assertEqual(get_dims(x), [])

  def test_get_dims_1d_tf(self):
    with self.test_session():
      x = tf.zeros([2])
      self.assertEqual(get_dims(x), [2])

  def test_get_dims_1d_np(self):
    with self.test_session():
      x = np.zeros([2])
      self.assertEqual(get_dims(x), [2])

  def test_get_dims_1d_rv(self):
    with self.test_session():
      x = Bernoulli(p=[0.5])
      self.assertEqual(get_dims(x), [1])

  def test_get_dims_2d_tf(self):
    with self.test_session():
      x = tf.zeros([2, 2])
      self.assertEqual(get_dims(x), [2, 2])

  def test_get_dims_2d_np(self):
    with self.test_session():
      x = np.zeros([2, 2])
      self.assertEqual(get_dims(x), [2, 2])

  def test_get_dims_2d_rv(self):
    with self.test_session():
      x = Bernoulli(p=[[0.5]])
      self.assertEqual(get_dims(x), [1, 1])

if __name__ == '__main__':
  tf.test.main()
