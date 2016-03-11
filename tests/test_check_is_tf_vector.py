from __future__ import print_function
import unittest
import tensorflow as tf
import numpy as np

from blackbox.util import check_is_tf_vector

class test_check_is_tf_vector(unittest.TestCase):

  def test_not_tf(self):
      x = np.array([1,2,3])
      with self.assertRaises(TypeError):
          check_is_tf_vector(x)

  def test_scalar(self):
      x = tf.constant(0.0)
      with self.assertRaises(TypeError):
          check_is_tf_vector(x)

  def test_1D_tensor_scalar(self):
      x = tf.zeros([1])
      with self.assertRaises(TypeError):
          check_is_tf_vector(x)

  def test_1D_tensor_vector(self):
      x = tf.zeros([2])
      check_is_tf_vector(x)

  def test_2D_tensor_vector(self):
      x = tf.zeros([2,1])
      check_is_tf_vector(x)

  def test_2D_tensor_matrix(self):
      x = tf.zeros([2,3])
      with self.assertRaises(TypeError):
          check_is_tf_vector(x)

  def test_3D_tensor(self):
      x = tf.zeros([2,1,3])
      with self.assertRaises(TypeError):
          check_is_tf_vector(x)                

if __name__ == '__main__':
    unittest.main()
