from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import get_variables


class test_get_variables_class(tf.test.TestCase):

  def test_v_structure(self):
    """a -> b -> e <- d <- c"""
    with self.test_session():
      a = tf.Variable(0.0)
      b = Normal(a, 1.0)
      c = tf.Variable(0.0)
      d = Normal(c, 1.0)
      e = Normal(b * d, 1.0)
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [a])
      self.assertEqual(get_variables(c), [])
      self.assertEqual(get_variables(d), [c])
      self.assertEqual(set(get_variables(e)), set([a, c]))

  def test_a_structure(self):
    """e <- d <- a -> b -> c"""
    with self.test_session():
      a = tf.Variable(0.0)
      b = Normal(a, 1.0)
      c = Normal(b, 1.0)
      d = Normal(a, 1.0)
      e = Normal(d, 1.0)
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [a])
      self.assertEqual(get_variables(c), [a])
      self.assertEqual(get_variables(d), [a])
      self.assertEqual(get_variables(e), [a])

  def test_chain_structure(self):
    """a -> b -> c -> d -> e"""
    with self.test_session():
      a = tf.Variable(0.0)
      b = tf.Variable(a)
      c = Normal(b, 1.0)
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [])
      self.assertEqual(get_variables(c), [b])

  def test_tensor(self):
    with self.test_session():
      a = tf.Variable(0.0)
      b = tf.constant(2.0)
      c = a + b
      d = tf.Variable(a)
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [])
      self.assertEqual(get_variables(c), [a])
      self.assertEqual(get_variables(d), [])

  def test_control_flow(self):
    with self.test_session():
      a = Bernoulli(0.5)
      b = tf.Variable(0.0)
      c = tf.constant(0.0)
      d = tf.cond(tf.cast(a, tf.bool), lambda: b, lambda: c)
      e = Normal(d, 1.0)
      self.assertEqual(get_variables(d), [b])
      self.assertEqual(get_variables(e), [b])

  def test_scan(self):
    with self.test_session():
      b = tf.Variable(0.0)
      op = tf.scan(lambda a, x: a + b + x, tf.constant([2.0, 3.0, 1.0]))

      self.assertEqual(get_variables(op), [b])

  def test_scan_with_a_structure(self):
    """copied from test_a_structure"""
    def cumsum(x):
      return tf.scan(lambda a, x: a + x, x)

    with self.test_session():
      a = tf.Variable([1.0, 1.0, 1.0])
      b = Normal(cumsum(a), tf.ones([3]))
      c = Normal(cumsum(b), tf.ones([3]))
      d = Normal(cumsum(a), tf.ones([3]))
      e = Normal(cumsum(d), tf.ones([3]))
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [a])
      self.assertEqual(get_variables(c), [a])
      self.assertEqual(get_variables(d), [a])
      self.assertEqual(get_variables(e), [a])

if __name__ == '__main__':
  tf.test.main()
