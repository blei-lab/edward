from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import get_parents


class test_get_parents_class(tf.test.TestCase):

  def test_v_structure(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=0.0, sigma=1.0)
      d = Normal(mu=c, sigma=1.0)
      e = Normal(mu=tf.multiply(b, d), sigma=1.0)
      self.assertEqual(get_parents(a), [])
      self.assertEqual(get_parents(b), [a])
      self.assertEqual(get_parents(c), [])
      self.assertEqual(get_parents(d), [c])
      self.assertEqual(set(get_parents(e)), set([b, d]))

  def test_a_structure(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=b, sigma=1.0)
      d = Normal(mu=a, sigma=1.0)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_parents(a), [])
      self.assertEqual(get_parents(b), [a])
      self.assertEqual(get_parents(c), [b])
      self.assertEqual(get_parents(d), [a])
      self.assertEqual(get_parents(e), [d])

  def test_chain_structure(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=b, sigma=1.0)
      d = Normal(mu=c, sigma=1.0)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_parents(a), [])
      self.assertEqual(get_parents(b), [a])
      self.assertEqual(get_parents(c), [b])
      self.assertEqual(get_parents(d), [c])
      self.assertEqual(get_parents(e), [d])

  def test_tensor(self):
    with self.test_session():
      a = Normal(mu=0.0, sigma=1.0)
      b = tf.constant(2.0)
      c = a + b
      d = Normal(mu=c, sigma=1.0)
      self.assertEqual(get_parents(a), [])
      self.assertEqual(get_parents(b), [])
      self.assertEqual(get_parents(c), [a])
      self.assertEqual(get_parents(d), [a])

  def test_control_flow(self):
    with self.test_session():
      a = Bernoulli(p=0.5)
      b = Normal(mu=0.0, sigma=1.0)
      c = tf.constant(0.0)
      d = tf.cond(tf.cast(a, tf.bool), lambda: b, lambda: c)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_parents(a), [])
      self.assertEqual(get_parents(b), [])
      self.assertEqual(get_parents(c), [])
      self.assertEqual(set(get_parents(d)), set([a, b]))
      self.assertEqual(set(get_parents(e)), set([a, b]))

if __name__ == '__main__':
  tf.test.main()
