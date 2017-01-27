from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import get_children


class test_get_children_class(tf.test.TestCase):

  def test_v_structure(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=0.0, sigma=1.0)
      d = Normal(mu=c, sigma=1.0)
      e = Normal(mu=tf.multiply(b, d), sigma=1.0)
      self.assertEqual(get_children(a), [b])
      self.assertEqual(get_children(b), [e])
      self.assertEqual(get_children(c), [d])
      self.assertEqual(get_children(d), [e])
      self.assertEqual(get_children(e), [])

  def test_a_structure(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=b, sigma=1.0)
      d = Normal(mu=a, sigma=1.0)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(set(get_children(a)), set([b, d]))
      self.assertEqual(get_children(b), [c])
      self.assertEqual(get_children(c), [])
      self.assertEqual(get_children(d), [e])
      self.assertEqual(get_children(e), [])

  def test_chain_structure(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=b, sigma=1.0)
      d = Normal(mu=c, sigma=1.0)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_children(a), [b])
      self.assertEqual(get_children(b), [c])
      self.assertEqual(get_children(c), [d])
      self.assertEqual(get_children(d), [e])
      self.assertEqual(get_children(e), [])

  def test_tensor(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = tf.constant(2.0)
      c = a + b
      d = Normal(mu=c, sigma=1.0)
      self.assertEqual(get_children(a), [d])
      self.assertEqual(get_children(b), [d])
      self.assertEqual(get_children(c), [d])
      self.assertEqual(get_children(d), [])

  def test_control_flow(self):
    with self.test_session():
      a = Bernoulli(p=0.5)
      b = Normal(mu=0.0, sigma=1.0)
      c = tf.constant(0.0)
      d = tf.cond(tf.cast(a, tf.bool), lambda: b, lambda: c)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_children(a), [e])
      self.assertEqual(get_children(b), [e])
      self.assertEqual(get_children(c), [e])
      self.assertEqual(get_children(d), [e])
      self.assertEqual(get_children(e), [])

if __name__ == '__main__':
  tf.test.main()
