from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import hessian


class test_hessian_class(tf.test.TestCase):

  def test_hessian_0d(self):
    with self.test_session():
      x1 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
      x2 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
      y = tf.pow(x1, tf.constant(2.0)) + tf.constant(2.0) * x1 * x2 + \
          tf.constant(3.0) * tf.pow(x2, tf.constant(2.0)) + \
          tf.constant(4.0) * x1 + tf.constant(5.0) * x2 + tf.constant(6.0)
      tf.global_variables_initializer().run()
      self.assertAllEqual(hessian(y, [x1]).eval(),
                          np.array([[2.0]]))
      self.assertAllEqual(hessian(y, [x2]).eval(),
                          np.array([[6.0]]))

  def test_hessian_1d(self):
    with self.test_session():
      x1 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
      x2 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
      y = tf.pow(x1, tf.constant(2.0)) + tf.constant(2.0) * x1 * x2 + \
          tf.constant(3.0) * tf.pow(x2, tf.constant(2.0)) + \
          tf.constant(4.0) * x1 + tf.constant(5.0) * x2 + tf.constant(6.0)
      x3 = tf.Variable(tf.random_normal([3], dtype=tf.float32))
      z = tf.pow(x2, tf.constant(2.0)) + tf.reduce_sum(x3)
      tf.global_variables_initializer().run()
      self.assertAllEqual(hessian(y, [x1, x2]).eval(),
                          np.array([[2.0, 2.0], [2.0, 6.0]]))
      self.assertAllEqual(hessian(z, [x3]).eval(),
                          np.zeros([3, 3]))
      self.assertAllEqual(hessian(z, [x2, x3]).eval(),
                          np.diag([2.0, 0.0, 0.0, 0.0]))

  def test_hessian_2d(self):
    with self.test_session():
      x1 = tf.Variable(tf.random_normal([3, 2], dtype=tf.float32))
      x2 = tf.Variable(tf.random_normal([2], dtype=tf.float32))
      y = tf.reduce_sum(tf.pow(x1, tf.constant(2.0))) + tf.reduce_sum(x2)
      tf.global_variables_initializer().run()
      self.assertAllEqual(hessian(y, [x1]).eval(),
                          np.diag([2.0] * 6))
      self.assertAllEqual(hessian(y, [x1, x2]).eval(),
                          np.diag([2.0] * 6 + [0.0] * 2))

  def test_all_finite_raises(self):
    with self.test_session():
      x1 = tf.Variable(np.nan * tf.random_normal([1], dtype=tf.float32))
      x2 = tf.Variable(tf.random_normal([1], dtype=tf.float32))
      y = tf.pow(x1, tf.constant(2.0)) + tf.constant(2.0) * x1 * x2 + \
          tf.constant(3.0) * tf.pow(x2, tf.constant(2.0)) + \
          tf.constant(4.0) * x1 + tf.constant(5.0) * x2 + tf.constant(6.0)
      tf.global_variables_initializer().run()
      with self.assertRaisesOpError('NaN'):
        hessian(y, [x1]).eval()
      with self.assertRaisesOpError('NaN'):
        hessian(y, [x1, x2]).eval()

if __name__ == '__main__':
  tf.test.main()
