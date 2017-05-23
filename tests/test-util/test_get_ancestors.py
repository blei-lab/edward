from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import get_ancestors


class test_get_ancestors_class(tf.test.TestCase):

  def test_v_structure(self):
    """a -> b -> e <- d <- c"""
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(a, 1.0)
      c = Normal(0.0, 1.0)
      d = Normal(c, 1.0)
      e = Normal(b * d, 1.0)
      self.assertEqual(get_ancestors(a), [])
      self.assertEqual(get_ancestors(b), [a])
      self.assertEqual(get_ancestors(c), [])
      self.assertEqual(get_ancestors(d), [c])
      self.assertEqual(set(get_ancestors(e)), set([a, b, c, d]))

  def test_a_structure(self):
    """e <- d <- a -> b -> c"""
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(a, 1.0)
      c = Normal(b, 1.0)
      d = Normal(a, 1.0)
      e = Normal(d, 1.0)
      self.assertEqual(get_ancestors(a), [])
      self.assertEqual(get_ancestors(b), [a])
      self.assertEqual(set(get_ancestors(c)), set([a, b]))
      self.assertEqual(get_ancestors(d), [a])
      self.assertEqual(set(get_ancestors(e)), set([a, d]))

  def test_chain_structure(self):
    """a -> b -> c -> d -> e"""
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(a, 1.0)
      c = Normal(b, 1.0)
      d = Normal(c, 1.0)
      e = Normal(d, 1.0)
      self.assertEqual(get_ancestors(a), [])
      self.assertEqual(get_ancestors(b), [a])
      self.assertEqual(set(get_ancestors(c)), set([a, b]))
      self.assertEqual(set(get_ancestors(d)), set([a, b, c]))
      self.assertEqual(set(get_ancestors(e)), set([a, b, c, d]))

  def test_tensor(self):
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = tf.constant(2.0)
      c = a + b
      d = Normal(c, 1.0)
      self.assertEqual(get_ancestors(a), [])
      self.assertEqual(get_ancestors(b), [])
      self.assertEqual(get_ancestors(c), [a])
      self.assertEqual(get_ancestors(d), [a])

  def test_control_flow(self):
    with self.test_session():
      a = Bernoulli(0.5)
      b = Normal(0.0, 1.0)
      c = tf.constant(0.0)
      d = tf.cond(tf.cast(a, tf.bool), lambda: b, lambda: c)
      e = Normal(d, 1.0)
      self.assertEqual(get_ancestors(a), [])
      self.assertEqual(get_ancestors(b), [])
      self.assertEqual(get_ancestors(c), [])
      self.assertEqual(set(get_ancestors(d)), set([a, b]))
      self.assertEqual(set(get_ancestors(e)), set([a, b]))

  def test_scan(self):
    """copied from test_chain_structure"""
    def cumsum(x):
      return tf.scan(lambda a, x: a + x, x)

    with self.test_session():
      a = Normal(tf.ones([3]), tf.ones([3]))
      b = Normal(cumsum(a), tf.ones([3]))
      c = Normal(cumsum(b), tf.ones([3]))
      d = Normal(cumsum(c), tf.ones([3]))
      e = Normal(cumsum(d), tf.ones([3]))
      self.assertEqual(get_ancestors(a), [])
      self.assertEqual(get_ancestors(b), [a])
      self.assertEqual(set(get_ancestors(c)), set([a, b]))
      self.assertEqual(set(get_ancestors(d)), set([a, b, c]))
      self.assertEqual(set(get_ancestors(e)), set([a, b, c, d]))

if __name__ == '__main__':
  tf.test.main()
