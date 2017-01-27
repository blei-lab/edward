from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import get_variables


class test_get_variables_class(tf.test.TestCase):

  def test_v_structure(self):
    with self.test_session():
      a = tf.Variable(0.0)
      b = Normal(mu=a, sigma=1.0)
      c = tf.Variable(0.0)
      d = Normal(mu=c, sigma=1.0)
      e = Normal(mu=tf.multiply(b, d), sigma=1.0)
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [a])
      self.assertEqual(get_variables(c), [])
      self.assertEqual(get_variables(d), [c])
      self.assertEqual(set(get_variables(e)), set([a, c]))

  def test_a_structure(self):
    with self.test_session():
      a = tf.Variable(0.0)
      b = Normal(mu=a, sigma=1.0)
      c = Normal(mu=b, sigma=1.0)
      d = Normal(mu=a, sigma=1.0)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_variables(a), [])
      self.assertEqual(get_variables(b), [a])
      self.assertEqual(get_variables(c), [a])
      self.assertEqual(get_variables(d), [a])
      self.assertEqual(get_variables(e), [a])

  def test_chain_structure(self):
    with self.test_session():
      a = tf.Variable(0.0)
      b = tf.Variable(a)
      c = Normal(mu=b, sigma=1.0)
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
      a = Bernoulli(p=0.5)
      b = tf.Variable(0.0)
      c = tf.constant(0.0)
      d = tf.cond(tf.cast(a, tf.bool), lambda: b, lambda: c)
      e = Normal(mu=d, sigma=1.0)
      self.assertEqual(get_variables(d), [b])
      self.assertEqual(get_variables(e), [b])

if __name__ == '__main__':
  tf.test.main()
